"""Pinhole helper with the exact camera convention of world.py (the executor), for fitting an SCM to a real clip.

  lift a detected box to the world (depth = distance along the camera's viewing direction, metres):
    python3 cam_helper.py lift --pos X,Y,Z --target X,Y,Z --fov 35 --box x0,y0,x1,y1 --depth D
      -> centre point, bottom-centre point (where an object stands), width and height in metres at that depth
  project a world point to the screen:
    python3 cam_helper.py project --pos X,Y,Z --target X,Y,Z --fov 35 --point X,Y,Z
      -> normalised screen position (x right, y down, 0..1) and the distance along the viewing direction
  find the depth at which a known size fills a box height:
    python3 cam_helper.py depth --fov 35 --box x0,y0,x1,y1 --height H
Boxes are normalised to the frame (x0, y0 top-left, x1, y1 bottom-right; the frame is 16:9).
"""
import os, argparse, math
import numpy as np

ASPECT = (lambda g: int(g[1]) / int(g[2]))(os.environ['CD_GEOM'].split(',')) if os.environ.get('CD_GEOM') else 1344 / 768  # clip aspect (geomrun.py)


def basis(pos, target):
    o = np.array(pos, float); fwd = np.array(target, float) - o; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    return o, fwd, right, up


def ray(pos, target, fov, u, v):
    o, fwd, right, up = basis(pos, target); t = math.tan(math.radians(fov) / 2)
    xx, yy = 2 * u - 1, 1 - 2 * v
    return o, fwd - xx * right * t * ASPECT + yy * up * t  # forward component 1: a point at depth z is o + z * ray


def vec(s):
    return [float(x) for x in s.split(',')]


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('lift', 'project'):
        p = sub.add_parser(name); p.add_argument('--pos', required=True); p.add_argument('--target', required=True)
        p.add_argument('--fov', type=float, default=35.0)
    sub.choices['lift'].add_argument('--box', required=True); sub.choices['lift'].add_argument('--depth', type=float, required=True)
    sub.choices['project'].add_argument('--point', required=True)
    p = sub.add_parser('depth'); p.add_argument('--fov', type=float, default=35.0); p.add_argument('--box', required=True)
    p.add_argument('--height', type=float, required=True)
    a = ap.parse_args()
    if a.cmd == 'lift':
        x0, y0, x1, y1 = vec(a.box); t = math.tan(math.radians(a.fov) / 2)
        o, r = ray(vec(a.pos), vec(a.target), a.fov, (x0 + x1) / 2, (y0 + y1) / 2)
        ob, rb = ray(vec(a.pos), vec(a.target), a.fov, (x0 + x1) / 2, y1)
        c, b = o + a.depth * r, ob + a.depth * rb
        print(f'centre        [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]')
        print(f'bottom-centre [{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}]   (ground contact if the object stands on y = {b[1]:.3f})')
        print(f'width {2 * t * ASPECT * a.depth * (x1 - x0):.3f} m, height {2 * t * a.depth * (y1 - y0):.3f} m at depth {a.depth} m')
    elif a.cmd == 'project':
        o, fwd, right, up = basis(vec(a.pos), vec(a.target)); t = math.tan(math.radians(a.fov) / 2)
        d = np.array(vec(a.point)) - o; z = d @ fwd
        xx = -(d @ right) / (z * t * ASPECT); yy = (d @ up) / (z * t)
        print(f'screen x {(xx + 1) / 2:.4f}, y {(1 - yy) / 2:.4f}, depth {z:.3f} m' + ('  (behind the camera)' if z <= 0 else ''))
    else:
        x0, y0, x1, y1 = vec(a.box); t = math.tan(math.radians(a.fov) / 2)
        print(f'depth {a.height / (2 * t * (y1 - y0)):.3f} m')


if __name__ == '__main__':
    main()
