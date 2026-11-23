"""Feature tracks for the measured camera (plan A). CPU only (node python with OpenCV).

    python camtrack.py LIST [--procs 32]

LIST: one job JSON per line: {"id", "video", "frames" (the anchor frames: the detector frames, which have a depth map), "depth_dir",
"mask_dir", "out"}. For every anchor a but the last, up to 1000 corners outside the entities' masks are tracked frame by frame
(pyramidal Lucas-Kanade, kept while the forward-backward error stays under 1 px) up to the next anchor b. Saved per anchor: their
pixel positions at a (p<a>, N x 2), their metric depth at a (z<a>, from the anchor's depth map), their positions in frames a+1..b
(t<a>, (b-a) x N x 2) with a valid flag (ok<a>), and their depth at b where still tracked (zb<a>, nan where lost; it ties the depth
scale of b to that of a). ws/cam_track.py turns these into camera poses for the planner's field of view and t=0 pose. When fewer
than 60 corners lie outside the masks, masked objects (not people, animals or vehicles) are used too.
"""
import argparse, json, sys, traceback
from multiprocessing import Pool
from pathlib import Path
import cv2
import numpy as np

LK = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
SELF_MOVING = ('human', 'animal', 'vehicle')


def read(video, n):
    cap = cv2.VideoCapture(video); out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    return out


def sample(dep, pts, W, H):
    """bilinear depth at full-resolution pixel positions (the depth map is W/4 x H/4)"""
    if not len(pts):   # every track of the segment was lost (fast motion, blur)
        return np.zeros(0, np.float32)
    h, w = dep.shape; x = (pts[:, 0] + .5) * w / W - .5; y = (pts[:, 1] + .5) * h / H - .5
    return cv2.remap(dep.astype(np.float32), x.astype(np.float32)[None], y.astype(np.float32)[None], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)[0]


def free_mask(mask_dir, a, types, shape, use_objects):
    m = np.full(shape, 255, np.uint8)
    for p in Path(mask_dir).glob(f'{a:03d}_*.png'):
        e = p.stem[4:]
        if use_objects and types.get(e, 'object') not in SELF_MOVING:
            continue
        k = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if k is not None:
            if k.shape != shape:
                k = cv2.resize(k, shape[::-1], interpolation=cv2.INTER_NEAREST)
            m[cv2.dilate((k > 127).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0] = 0
    return m


def one(job):
    try:
        anchors = sorted(job['frames']); gray = read(job['video'], anchors[-1] + 1); H, W = gray[0].shape
        anchors = [a for a in anchors if a < len(gray)]; out = dict(W=W, H=H, F=len(gray), anchors=np.array(anchors)); stats = {}
        for a, b in zip(anchors, anchors[1:]):
            dep = np.load(Path(job['depth_dir']) / f'{a:03d}.npy').astype(np.float32)
            p = None
            for use_objects in (False, True):
                p = cv2.goodFeaturesToTrack(gray[a], 1000, 0.005, 8, mask=free_mask(job['mask_dir'], a, job.get('types', {}), gray[a].shape, use_objects))
                if p is not None and len(p) >= 60:
                    break
            if p is None:
                p = np.zeros((0, 1, 2), np.float32)
            p0 = p.reshape(-1, 2); cur = p0.copy(); alive = np.ones(len(p0), bool); tr, ok = [], []
            for i in range(a + 1, b + 1):
                if len(p0):
                    nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray[i - 1], gray[i], cur.reshape(-1, 1, 2), None, **LK)
                    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray[i], gray[i - 1], nxt, None, **LK)
                    fb = np.linalg.norm(back.reshape(-1, 2) - cur, axis=1)
                    nxt = nxt.reshape(-1, 2)
                    inside = (nxt[:, 0] >= 0) & (nxt[:, 0] < W) & (nxt[:, 1] >= 0) & (nxt[:, 1] < H)
                    alive &= (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0) & inside
                    cur = np.where(alive[:, None], nxt, cur)
                tr.append(cur.copy()); ok.append(alive.copy())
            z = sample(dep, p0, W, H) if len(p0) else np.zeros(0, np.float32)
            zb = np.full(len(p0), np.nan, np.float32)
            db = Path(job['depth_dir']) / f'{b:03d}.npy'
            if db.exists() and len(p0):
                zb[alive] = sample(np.load(db).astype(np.float32), cur[alive], W, H)
            out.update({f'p{a}': p0.astype(np.float32), f'z{a}': z.astype(np.float32), f't{a}': np.array(tr, np.float32),
                        f'ok{a}': np.array(ok, bool), f'zb{a}': zb})
            stats[a] = [int(len(p0)), int(alive.sum())]
        Path(job['out']).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(job['out']).with_suffix('.tmp.npz'); np.savez_compressed(tmp, **out); tmp.rename(job['out'])
        return dict(id=job['id'], status='pass', points=stats)
    except Exception as exc:
        return dict(id=job['id'], status='failed', error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()[-800:])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('list'); ap.add_argument('--procs', type=int, default=32)
    a = ap.parse_args(); cv2.setNumThreads(1)
    jobs = [json.loads(Path(l.strip()).read_text()) for l in Path(a.list).read_text().splitlines() if l.strip()]
    bad = 0
    with Pool(min(a.procs, max(1, len(jobs)))) as pool:
        for r in pool.imap_unordered(one, jobs):
            bad += r['status'] != 'pass'; print(json.dumps(r), flush=True)
    print('CAMTRACK_DONE', json.dumps(dict(total=len(jobs), failed=bad)), flush=True)
    sys.exit(1 if bad == len(jobs) else 0)


if __name__ == '__main__':
    main()
