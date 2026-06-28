"""Vector drawing of schematic component glyphs.

Every glyph is drawn between two terminal points ``t1`` and ``t2`` that lie on
the wire axis. The glyph body (and its leads) are drawn on the *image only*; the
caller is responsible for drawing the connecting wires (node -> t1 and t2 ->
node) onto both the image and the wire mask. This keeps the U-Net target mask
strictly "wires only", which is what inference expects.

Each function returns the glyph bounding box (x1, y1, x2, y2) in pixels for the
YOLO label.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

BLACK = (0, 0, 0)


def _unit(t1: np.ndarray, t2: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (axis unit vector, perpendicular unit vector, length)."""
    d = t2 - t1
    length = float(np.hypot(*d)) + 1e-6
    u = d / length
    perp = np.array([-u[1], u[0]])
    return u, perp, length


def _bbox(points: list[np.ndarray], pad: int = 4) -> tuple[int, int, int, int]:
    pts = np.array(points)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return int(x1 - pad), int(y1 - pad), int(x2 + pad), int(y2 + pad)


def _line(img, a, b, thickness):
    cv2.line(img, (int(round(a[0])), int(round(a[1]))),
             (int(round(b[0])), int(round(b[1]))), BLACK, thickness, cv2.LINE_AA)


def resistor(img, t1, t2, thickness, scale):
    """ANSI zig-zag resistor between t1 and t2."""
    u, perp, length = _unit(t1, t2)
    amp = 7 * scale
    n = 6
    body_start = t1 + u * (length * 0.15)
    body_len = length * 0.70
    pts = [body_start]
    for i in range(1, n):
        along = body_start + u * (body_len * i / n)
        off = perp * (amp if i % 2 else -amp)
        pts.append(along + off)
    pts.append(t1 + u * (length * 0.85))
    # leads
    _line(img, t1, pts[0], thickness)
    _line(img, pts[-1], t2, thickness)
    for a, b in zip(pts[:-1], pts[1:]):
        _line(img, a, b, thickness)
    return _bbox(pts)


def capacitor(img, t1, t2, thickness, scale):
    """Two parallel plates with a gap."""
    u, perp, length = _unit(t1, t2)
    half_plate = 11 * scale
    gap = 7 * scale
    mid = (t1 + t2) / 2
    p1 = mid - u * (gap / 2)
    p2 = mid + u * (gap / 2)
    a1, a2 = p1 + perp * half_plate, p1 - perp * half_plate
    b1, b2 = p2 + perp * half_plate, p2 - perp * half_plate
    _line(img, t1, p1, thickness)   # lead
    _line(img, t2, p2, thickness)   # lead
    _line(img, a1, a2, thickness)   # plate 1
    _line(img, b1, b2, thickness)   # plate 2
    return _bbox([a1, a2, b1, b2])


def inductor(img, t1, t2, thickness, scale):
    """Series of semicircular bumps."""
    u, perp, length = _unit(t1, t2)
    n = 4
    body_start = t1 + u * (length * 0.15)
    body_len = length * 0.70
    r = body_len / (2 * n)
    angle = math.degrees(math.atan2(u[1], u[0]))
    pts = [body_start, t2]
    for i in range(n):
        center = body_start + u * (r * (2 * i + 1))
        c = (int(round(center[0])), int(round(center[1])))
        cv2.ellipse(img, c, (int(r), int(8 * scale)), angle, 180, 360, BLACK, thickness, cv2.LINE_AA)
        pts.append(center + perp * (8 * scale))
    _line(img, t1, body_start, thickness)
    _line(img, body_start + u * body_len, t2, thickness)
    return _bbox(pts)


def diode(img, t1, t2, thickness, scale):
    """Triangle pointing t1->t2 with a cathode bar."""
    u, perp, length = _unit(t1, t2)
    h = 10 * scale
    apex = (t1 + t2) / 2 + u * (6 * scale)
    base_c = (t1 + t2) / 2 - u * (6 * scale)
    b1, b2 = base_c + perp * h, base_c - perp * h
    bar1, bar2 = apex + perp * h, apex - perp * h
    _line(img, t1, base_c, thickness)
    _line(img, apex, t2, thickness)
    cv2.line(img, tuple(b1.astype(int)), tuple(b2.astype(int)), BLACK, thickness, cv2.LINE_AA)
    cv2.line(img, tuple(b1.astype(int)), tuple(apex.astype(int)), BLACK, thickness, cv2.LINE_AA)
    cv2.line(img, tuple(b2.astype(int)), tuple(apex.astype(int)), BLACK, thickness, cv2.LINE_AA)
    cv2.line(img, tuple(bar1.astype(int)), tuple(bar2.astype(int)), BLACK, thickness, cv2.LINE_AA)
    return _bbox([b1, b2, bar1, bar2])


def _circle_source(img, t1, t2, scale, symbol):
    u, perp, length = _unit(t1, t2)
    mid = (t1 + t2) / 2
    r = int(13 * scale)
    c = (int(round(mid[0])), int(round(mid[1])))
    cv2.circle(img, c, r, BLACK, max(1, scale), cv2.LINE_AA)
    _line(img, t1, mid - u * r, max(1, scale))
    _line(img, mid + u * r, t2, max(1, scale))
    if symbol == "battery":
        # + and - signs along the axis inside the circle
        plus = mid - u * (r * 0.4)
        minus = mid + u * (r * 0.4)
        cv2.line(img, tuple((plus - perp * 4).astype(int)), tuple((plus + perp * 4).astype(int)), BLACK, 1)
        cv2.line(img, tuple((plus - u * 4).astype(int)), tuple((plus + u * 4).astype(int)), BLACK, 1)
        cv2.line(img, tuple((minus - perp * 4).astype(int)), tuple((minus + perp * 4).astype(int)), BLACK, 1)
    elif symbol == "current":
        # arrow along axis
        tip = mid + u * (r * 0.5)
        tail = mid - u * (r * 0.5)
        cv2.arrowedLine(img, tuple(tail.astype(int)), tuple(tip.astype(int)), BLACK, max(1, scale), tipLength=0.4)
    corner = np.array([r, r])
    return _bbox([mid - corner, mid + corner], pad=2)


def voltage_source(img, t1, t2, thickness, scale):
    return _circle_source(img, t1, t2, scale, "battery")


def current_source(img, t1, t2, thickness, scale):
    return _circle_source(img, t1, t2, scale, "current")


def ground(img, node, direction, thickness, scale):
    """Single-terminal ground symbol drawn from ``node`` along ``direction``.

    The connecting stub (node -> symbol top) is drawn here on the image; the
    caller draws the same stub on the wire mask.
    """
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    perp = np.array([-direction[1], direction[0]])
    top = node + direction * (14 * scale)
    _line(img, node, top, thickness)
    widths = [14, 9, 4]
    pts = [top]
    for i, w in enumerate(widths):
        center = top + direction * (5 * scale * (i + 1))
        a, b = center + perp * w * scale * 0.5, center - perp * w * scale * 0.5
        _line(img, a, b, thickness)
        pts.extend([a, b])
    return _bbox(pts), top  # box, and the stub endpoint (for the wire mask)


# Maps a class name to (draw fn, terminal count). Single-terminal symbols
# (ground) are handled specially by the generator.
TWO_TERMINAL = {
    "resistor": resistor,
    "capacitor": capacitor,
    "inductor": inductor,
    "diode": diode,
    "voltage_source": voltage_source,
    "current_source": current_source,
}
