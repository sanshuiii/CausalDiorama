"""Shared parts of the F4 edit (inv_fix_v4.py) on MiniMax-H3 Ref2VA + CWM LoRA: conditions from an h3_config.json of the stage-2 kind
(anchor locked as frame 0), the patched sample_joint_av step re-implemented one step at a time (Sampler: velocity, step size, sampling
with a replayed audio trajectory), the rank-0 video VAE (Pixels), video reading / writing and pixel metrics. The DiT stays resident
(tools/cwm/resident.py). CD_GEOM="frames,W,H" selects a clip geometry other than 124 frames at 1344x768 (flex_geom.py).
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

sys.path.insert(0, '@CD_SHARED@/tools/cwm')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import flex_geom  # noqa: E402  (CD_GEOM="frames,W,H": a clip geometry other than 124 frames at 1344x768)
flex_geom.apply_from_env()
import resident as R  # noqa: E402,F401  (puts the acceleration code on sys.path; used by inv_fix_v4)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from optimizations import rank, world  # noqa: E402,F401  (used by inv_fix_v4)

FRAMES, W, H = flex_geom.active(); FPS = 24
LATENT = flex_geom.derived(FRAMES, W, H)['VIDEO_LATENT_SHAPE']
REL = '@CD_NODE@/cwm_h3_reproduction_v1/release/models'
VAE = f'{REL}/vae/minimax_h3_video_vae_fp16.safetensors'


def log(**kw):
    if rank() == 0:
        print(json.dumps(kw), flush=True)


# ---- conditions -------------------------------------------------------------------------------------------------------------
def cwm_inputs(path, device):
    import multiref
    from cwm_h3_inference.config import load_config
    config = load_config(pathlib.Path(path), check_files=True)
    assert config.steps == 20 and len(config.windows) == 1
    multiref.validate_cache(config)
    inputs = multiref.build_inputs(config, device)
    key = (str(config.models.dit), str(config.models.lora), config.expected_lora_modules)
    return inputs, key, config


class Sampler:
    """The patched sample_joint_av, one step at a time, bound to one set of conditions."""

    def __init__(self, res, inputs, job_id):
        from musubi_tuner.minimax_h3.sampling import build_shifted_schedule
        self.res, self.x = res, inputs
        self.steps = inputs['steps']
        self.mask, self.fixed = inputs.get('video_denoise_mask'), inputs.get('input_video_latents')
        res.prepare_adaln(job_id, inputs['layout'], inputs['text_token_tags'], self.mask, inputs['initial_video'].shape, self.steps,
                          True, 0.0)
        s = build_shifted_schedule(self.steps, video_shift=inputs['video_shift'], audio_shift=inputs['audio_shift'],
                                   device=inputs['initial_video'].device)
        self.sv, self.sa = s.video, s.audio

    def velocity(self, i, video, audio):
        st = self.res.state
        st.step = i - 1  # RuntimeState.begin advances the cursor to i and checks the model times against the AdaLN tables
        st.skip = False
        sigma_v, sigma_a = self.sv[i].to(dtype=torch.float32), self.sa[i].to(dtype=torch.float32)
        model_video = video if self.mask is None else torch.where(self.mask, video, self.fixed)
        kw = {} if self.mask is None else {'video_denoise_mask': self.mask}
        x = self.x
        p = self.res.model(video_latents=model_video, audio_latents=audio, text_hidden_states=x['text_hidden_states'],
                           text_token_tags=x['text_token_tags'], layout=x['layout'], model_t_video=1.0 - sigma_v,
                           model_t_audio=1.0 - sigma_a, visual_condition_latents=x['visual_condition_latents'],
                           audio_condition_latents=x['audio_condition_latents'], visual_condition_clean=x['visual_condition_clean'],
                           audio_condition_clean=x['audio_condition_clean'], **kw)
        pv = p.video
        if self.mask is not None:
            pv = torch.where(self.mask, pv, (self.fixed - video) / sigma_v.to(video))
        return pv, p.audio

    def delta(self, i, video, audio):
        return (self.sv[i] - self.sv[i + 1]).to(video), (self.sa[i] - self.sa[i + 1]).to(audio)

    @torch.inference_mode()
    def sample(self, video, audio, callback=None, audio_traj=None, record=False):
        """audio_traj: the audio state at every step (21 entries) to replay instead of integrating the audio (audio is not studied:
        replaying one recorded trajectory keeps it identical in every run of a job). record=True also returns the audio trajectory."""
        traj = [audio]
        for i in range(self.steps):
            a_i = audio if audio_traj is None else audio_traj[i]
            pv, pa = self.velocity(i, video, a_i)
            dv, da = self.delta(i, video, a_i)
            video = video + dv * pv
            audio = a_i + da * pa if audio_traj is None else audio_traj[i + 1]
            traj.append(audio)
            if callback:
                callback(i)
        return (video, audio, traj) if record else (video, audio)

    def region(self, t):
        return t if self.mask is None else t[self.mask.expand_as(t)]


# ---- pixels -----------------------------------------------------------------------------------------------------------------
def read_video(path):
    import av
    frames = []
    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            frames.append(torch.from_numpy(f.to_ndarray(format='rgb24')))
    x = torch.stack(frames)
    assert tuple(x.shape[1:]) == (H, W, 3) and x.shape[0] >= FRAMES, (path, tuple(x.shape))
    return x[:FRAMES]


def write_video(pixels, path):
    import av
    path = pathlib.Path(path); tmp = path.with_suffix('.tmp.mp4')
    with av.open(str(tmp), 'w', options={'movflags': '+faststart'}) as c:
        s = c.add_stream('libx264', rate=FPS); s.width, s.height, s.pix_fmt = W, H, 'yuv420p'
        s.options = {'crf': '12'}
        for px in pixels:
            for pk in s.encode(av.VideoFrame.from_ndarray(px.numpy(), format='rgb24')):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    tmp.replace(path)


class Pixels:
    """Rank-0 VAE: encode (posterior mean, as vae.encode) and decode to uint8 frames [F,H,W,3]."""

    def __init__(self, device):
        from musubi_tuner.minimax_h3.video_vae import load_video_vae
        self.device = device
        self.vae = load_video_vae(VAE, device=device, dtype=torch.float16, disable_mmap=False).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, frames):
        from musubi_tuner.minimax_h3_generate_video import _prepare_pixels
        return self.vae.encode(_prepare_pixels(frames).to(self.device, torch.float16)).float()

    @torch.inference_mode()
    def decode(self, latent):
        d = self.vae.decode(latent.to(self.device, torch.float16))[0, :, :FRAMES].float().clamp(-1, 1)
        return ((d + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().contiguous()


def _gauss(device, size=11, sigma=1.5):
    g = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


@torch.inference_mode()
def pixel_metrics(a, b, device):
    """a, b uint8 [F,H,W,3] -> PSNR (RGB) and grey-level SSIM (11x11 Gaussian window), per frame and mean."""
    k = _gauss(device); ps, ss = [], []
    for i in range(a.shape[0]):
        x = a[i].to(device).float(); y = b[i].to(device).float()
        mse = ((x - y) ** 2).mean().item(); ps.append(99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse))
        gx = (x.double() @ torch.tensor([.299, .587, .114], device=device, dtype=torch.float64))[None, None]
        gy = (y.double() @ torch.tensor([.299, .587, .114], device=device, dtype=torch.float64))[None, None]
        kd = k.double(); mx, my = F.conv2d(gx, kd), F.conv2d(gy, kd)
        vx = (F.conv2d(gx * gx, kd) - mx ** 2).clamp_min(0); vy = (F.conv2d(gy * gy, kd) - my ** 2).clamp_min(0)
        cxy = F.conv2d(gx * gy, kd) - mx * my  # float64 + clamped variances (float32 gave SSIM slightly above 1)
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        ss.append((((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))).mean().item())
    return dict(psnr=sum(ps) / len(ps), ssim=sum(ss) / len(ss), psnr_frames=[round(p, 3) for p in ps],
                ssim_frames=[round(s, 5) for s in ss], psnr_from1=sum(ps[1:]) / (len(ps) - 1), ssim_from1=sum(ss[1:]) / (len(ss) - 1))


# ---- jobs -------------------------------------------------------------------------------------------------------------------
