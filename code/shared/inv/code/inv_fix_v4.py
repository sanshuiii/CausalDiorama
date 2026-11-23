"""F4 inversion edit of a real video (editing): one torchrun process (8 GPUs), H3 Ref2VA + CWM LoRA loaded once, cases one after another.

    torchrun --standalone --nproc_per_node=8 inv_fix_v4.py --cases CASES.json [--f4-skip-recon]

CASES.json: [{"id", "cond_a", "cond_b", "source", "out_dir", "state" (optional: state.pt of an earlier run whose 20-step audio
             trajectory is reused), "compare": {name: mp4}}]
A = the fitted plan's conditions (cond_a, an h3_config.json of the stage-2 kind with the real frame 0 locked), B = the edited plan's
conditions (cond_b, plan B's first frame locked). Audio is not edited: one 20-step audio trajectory, recorded from the seed with
conditions A, is replayed in every pass.
F4 (noise anchoring with recorded offsets): x_i = (1 - sigma_i) X + sigma_i S on a straight line from the source latent X to one
Gaussian S (the seed's initial noise), offsets o_i = x_{i+1} - x_i - d_i v_A(x_i) along the 20-step schedule; the edit integrates
x <- x + d_i v_B(x) + o_i from S; the reconstruction (skipped with --f4-skip-recon, as in the runs) integrates x <- x + d_i v_A(x) + o_i.
Outputs per case: F4_edit.mp4 (and F4_recon.mp4), metrics.json (evaluations, offset sizes, pixel metrics against the source).
"""
from __future__ import annotations

import argparse, json, pathlib, sys, time, traceback
sys.path.insert(0, '@CD_SHARED@/tools/cwm')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # the deployed code directory (inv_run, flex_geom)
import inv_run as IR  # noqa: E402  (Sampler, inputs, VAE pixels, metrics; imports the resident runtime)
import torch  # noqa: E402

R, rank, world = IR.R, IR.rank, IR.world


def rel(a, b, region):
    return float(region(a).norm() / region(b).norm().clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cases', type=pathlib.Path, required=True)
    ap.add_argument('--f4-skip-recon', action='store_true', help='skip the reconstruction pass (20 evaluations; the edit does not depend on it)')
    a = ap.parse_args()
    cases = json.loads(a.cases.read_text()); meth = 'F4'
    device = R.setup(); res = R.Resident(device); pix = [None]

    def ensure(key, config):
        if res.model is None:
            from loading import load_broadcast
            res.load(key, lambda: load_broadcast(config, device), 'fix')
        else:
            res.check_key(key)

    def pixels():
        if pix[0] is None:
            pix[0] = IR.Pixels(device)
        return pix[0]

    for c in cases:
        out = pathlib.Path(c['out_dir']); mfile = out / 'metrics.json'
        M = json.loads(mfile.read_text()) if mfile.exists() else dict(id=c['id'], methods={})
        if M['methods'].get(meth, {}).get('status') == 'pass':
            IR.log(case=c['id'], method=meth, status='skipped'); continue
        t0 = time.monotonic(); evals = [0]; rec = dict(status='running'); refs = {}
        if rank() == 0:
            out.mkdir(parents=True, exist_ok=True)

        def save():
            if rank() == 0:
                out.mkdir(parents=True, exist_ok=True); M['methods'][meth] = rec; mfile.write_text(json.dumps(M, indent=1))

        def decode(name, lat):
            def go():
                frames = pixels().decode(lat)
                IR.write_video(frames, out / f'{meth}_{name}.mp4')
                rec.setdefault('pixels', {})[name] = {k: IR.pixel_metrics(frames, v, device) for k, v in refs.items()}
                torch.cuda.empty_cache()
            R.on_rank0(device, go, f'decode {meth} {name}')

        try:
            ia, key, cfg = IR.cwm_inputs(c['cond_a'], device); ensure(key, cfg)
            ib, _, _ = IR.cwm_inputs(c['cond_b'], device)
            box = torch.empty(list(IR.LATENT), device=device)

            def enc():
                refs['source'] = IR.read_video(c['source'])
                for name, p in c.get('compare', {}).items():
                    refs[name] = IR.read_video(p)
                box.copy_(pixels().encode(refs['source']))
            R.on_rank0(device, enc, 'encode source')
            if world() > 1:
                torch.distributed.broadcast(box, 0)
            if c.get('state') and pathlib.Path(c['state']).exists():
                atraj = torch.load(c['state'], map_location=device)['atraj']
            else:   # the 20-step audio trajectory from the seed with conditions A (20 evaluations)
                _, _, atraj = IR.Sampler(res, ia, c['id'] + ':A20seed').sample(ia['initial_video'], ia['initial_audio'], record=True)
            X = box if ia.get('video_denoise_mask') is None else torch.where(ia['video_denoise_mask'], box, ia['input_video_latents'])
            sa = IR.Sampler(res, ia, c['id'] + ':A20'); S_ = ia['initial_video']
            xs = [(1 - float(sa.sv[i])) * X + float(sa.sv[i]) * S_ for i in range(21)]; xs[20] = X; offs = []
            for i in range(20):
                pv, _ = sa.velocity(i, xs[i], atraj[i]); dv, _ = sa.delta(i, xs[i], atraj[i]); offs.append(xs[i + 1] - xs[i] - dv * pv)
            evals[0] += 20
            rec['inversion'] = dict(evaluations=20, offset_rel=[round(rel(o, X, sa.region), 4) for o in offs])
            if not a.f4_skip_recon:
                v = S_
                for i in range(20):
                    pv, _ = sa.velocity(i, v, atraj[i]); dv, _ = sa.delta(i, v, atraj[i]); v = v + dv * pv + offs[i]
                evals[0] += 20; decode('recon', v)
            sb = IR.Sampler(res, ib, c['id'] + ':B20F4'); v = S_
            for i in range(20):
                pv, _ = sb.velocity(i, v, atraj[i]); dv, _ = sb.delta(i, v, atraj[i]); v = v + dv * pv + offs[i]
            evals[0] += 20; decode('edit', v)
            rec.update(status='pass', evaluations=evals[0], seconds=round(time.monotonic() - t0, 1))
        except Exception as exc:
            rec.update(status='failed', error=str(exc)[-2000:], traceback=traceback.format_exc())
            IR.log(case=c['id'], method=meth, status='failed', error=str(exc)[-1500:])
            if isinstance(exc, R.JobFailed) and res.load_failed:
                save(); R.finish(); return
        save(); IR.log(case=c['id'], method=meth, status=rec['status'], seconds=rec.get('seconds'))
    R.finish()


if __name__ == '__main__':
    main()
