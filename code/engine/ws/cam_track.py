"""Measured camera for the fitting planner (plan A): the clip's real camera motion as the SCM camera node C.

    python3 geomrun.py FRAMES,W,H cam_track.py TRACKS.npz --pos X,Y,Z --target X,Y,Z --fov F [--every 4]

TRACKS.npz comes from shared/inv/code/camtrack.py: background corners tracked between the detector frames, with their metric monocular depth.
For the given t=0 pose and field of view (the executor's pinhole, as cam_helper.py) it solves the camera of every frame against the
tracks (robust reprojection fit; the depth scale of each later anchor is tied to frame 0's through the tracked points), then writes C:
`fixed` when the camera moves less than 0.3 degrees and 3 cm, else `keyframes` every --every frames (and the last frame), interp
linear, as position + target in the plan's world (+X screen right, +Y up; the target keeps the t=0 distance). Roll and zoom are not
modelled (world.py has neither roll nor a measured zoom here): a zoom comes out as a move along the view. Prints the summary, C as one
JSON line, and the camera at every detector frame (for cam_helper.py lift / project on that frame).
"""
import argparse, json, math, os
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

FPS = 24


def vec(s):
    return [float(x) for x in s.split(',')]


def basis(pos, target):
    """world.py's camera axes: forward, its `right` (= screen left) and up"""
    o = np.array(pos, float); fwd = np.array(target, float) - o; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    return o, fwd, right, up


def project(x, R, t, K):
    c = x @ R.T + t
    return np.stack([K[0] * c[:, 0] / c[:, 2] + K[2], K[1] * c[:, 1] / c[:, 2] + K[3]], 1), c[:, 2]


def solve(X, obs, guess, K):
    """pose (rotation vector, translation) taking anchor-camera points X to the frame where they are seen at obs"""
    def res(p):
        uv, z = project(X, Rotation.from_rotvec(p[:3]).as_matrix(), p[3:], K)
        return np.where(z[:, None] > 1e-3, uv - obs, 50.).ravel()
    p = least_squares(res, guess, loss='soft_l1', f_scale=1.0, x_scale='jac').x
    e = np.linalg.norm(res(p).reshape(-1, 2), axis=1); keep = e < max(3.0, 3 * np.median(e))
    if keep.sum() >= 8 and keep.sum() < len(e):   # refit without the outliers (things that move on their own)
        X, obs = X[keep], obs[keep]
        p = least_squares(res, p, loss='soft_l1', f_scale=1.0, x_scale='jac').x
        e = np.linalg.norm(res(p).reshape(-1, 2), axis=1)
    return p, float(np.median(e)), len(X)


def track(npz, fov):
    d = np.load(npz); W, H, F = int(d['W']), int(d['H']), int(d['F']); anchors = [int(a) for a in d['anchors']]
    f = (H / 2) / math.tan(math.radians(fov) / 2); K = (f, f, W / 2 - .5, H / 2 - .5)   # pixel centres at integers (OpenCV)
    R0 = {0: np.eye(3)}; T0 = {0: np.zeros(3)}; err, used = {}, {}
    scale = 1.0
    for a, b in zip(anchors, anchors[1:]):
        p0, z, tr, ok, zb = d[f'p{a}'], d[f'z{a}'], d[f't{a}'], d[f'ok{a}'], d[f'zb{a}']
        good = np.isfinite(z) & (z > 0)
        X = np.stack([(p0[:, 0] - K[2]) / K[0], (p0[:, 1] - K[3]) / K[1], np.ones(len(p0))], 1) * (scale * z)[:, None]
        guess = np.zeros(6); Ra, ta = R0[a], T0[a]
        for k, i in enumerate(range(a + 1, b + 1)):
            m = good & ok[k]
            if m.sum() >= 8:
                guess, err[i], used[i] = solve(X[m], tr[k][m], guess, K)
            else:
                err[i], used[i] = float('nan'), int(m.sum())   # too few tracks: the camera holds its last relative pose
            R = Rotation.from_rotvec(guess[:3]).as_matrix(); t = guess[3:]
            R0[i] = R @ Ra; T0[i] = R @ ta + t
        m = good & ok[-1] & np.isfinite(zb) & (zb > 0)
        if m.sum() >= 20:   # depth scale of b in frame 0's terms: predicted depth of the tracked points / b's own depth
            zp = (X[m] @ Rotation.from_rotvec(guess[:3]).as_matrix().T + guess[3:])[:, 2]
            scale = float(np.median(zp / zb[m]))
    return F, R0, T0, err, used


def smooth(a, w=5):
    out = np.empty_like(a)
    for i in range(len(a)):
        lo, hi = max(0, i - w // 2), min(len(a), i + w // 2 + 1); out[i] = a[lo:hi].mean(0)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('tracks'); ap.add_argument('--pos', required=True); ap.add_argument('--target', required=True)
    ap.add_argument('--fov', type=float, required=True); ap.add_argument('--every', type=int, default=4)
    a = ap.parse_args(); pos, tgt = vec(a.pos), vec(a.target)
    F, R0, T0, err, used = track(a.tracks, a.fov)
    if 'CD_GEOM' in os.environ:
        F = min(F, int(os.environ['CD_GEOM'].split(',')[0]))
    frames = [i for i in range(F) if i in R0]
    o, fwd, right, up = basis(pos, tgt); B = np.stack([-right, -up, fwd], 1)   # frame-0 camera axes (x right, y down, z forward) in the world
    dist = float(np.linalg.norm(np.array(tgt) - np.array(pos)))
    P = np.array([o + B @ (-R0[i].T @ T0[i]) for i in frames]); V = np.array([B @ (R0[i].T @ [0, 0, 1.]) for i in frames])
    P, V = smooth(P), smooth(V); V /= np.linalg.norm(V, axis=1, keepdims=True); P[0], V[0] = o, fwd
    ang = np.degrees(np.arccos(np.clip(V @ fwd, -1, 1))); mv = np.linalg.norm(P - o, axis=1)
    yaw = np.degrees(np.arctan2(V[:, 0], V[:, 2]) - math.atan2(fwd[0], fwd[2])); pitch = np.degrees(np.arcsin(np.clip(V[:, 1], -1, 1)) - math.asin(fwd[1]))
    e = np.array([err[i] for i in frames[1:] if np.isfinite(err.get(i, np.nan))])
    still = ang.max() < 0.3 and mv.max() < 0.03
    C = {'id': 'camera', 'controller': 'fixed' if still else 'keyframes', 'position': [round(x, 3) for x in pos], 'target': [round(x, 3) for x in tgt],
         'vertical_fov_deg': a.fov}
    if not still:
        keys = sorted(set(range(a.every, len(frames), a.every)) | {len(frames) - 1})
        C['interp'] = 'linear'
        C['keyframes'] = [[round(frames[k] / FPS, 4), {'position': [round(x, 3) for x in P[k]], 'target': [round(x, 3) for x in P[k] + V[k] * dist]}] for k in keys]
    lost = [i for i in frames[1:] if not np.isfinite(err.get(i, np.nan))]
    print(f"measured camera over {len(frames)} frames: {'still' if still else 'moving'}; view turns up to {ang.max():.1f} deg "
          f"(at the end: pan {yaw[-1]:+.1f} deg, + = right; tilt {pitch[-1]:+.1f} deg, + = up), moves up to {mv.max():.2f} m "
          f"(at the end {P[-1, 0] - o[0]:+.2f} right, {P[-1, 1] - o[1]:+.2f} up, {float((P[-1] - o) @ fwd):+.2f} forward along the t=0 view); "
          f"fit: median reprojection error {np.median(e) if len(e) else float('nan'):.2f} px, {int(np.median([used[i] for i in frames[1:]])) if len(frames) > 1 else 0} tracks a frame"
          + (f"; no fit (too few tracks) at frames {lost[:8]}{'...' if len(lost) > 8 else ''}: the camera holds there" if lost else ''))
    print('C (copy into the plan unchanged, then add "motion"):')
    print(json.dumps(C))
    print('camera at the detector frames (use these with cam_helper.py lift / project on that frame):')
    for i in sorted(set(range(0, len(frames), 12)) | {len(frames) - 1}):
        p, t = P[i], P[i] + V[i] * dist
        print(f"  frame {frames[i]:3d}: --pos {p[0]:.3f},{p[1]:.3f},{p[2]:.3f} --target {t[0]:.3f},{t[1]:.3f},{t[2]:.3f} --fov {a.fov:g}")


if __name__ == '__main__':
    main()
