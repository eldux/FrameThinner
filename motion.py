"""Feature extraction and motion checks used in the Nienaber review."""
import math
import cv2
import numpy as np

def features(path, detector, upper_scene=False, excluded_boxes=(), scale=1.0, mask_sky=True, regions=None):
    if scale > 1:
        mode = cv2.IMREAD_COLOR if upper_scene else cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_REDUCED_COLOR_2 if upper_scene else cv2.IMREAD_REDUCED_GRAYSCALE_2
    pixels = cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)
    if pixels is None:
        return None
    size = (round(640*scale), round(480*scale))
    pixels = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY) if upper_scene else pixels
    mask = np.zeros((480, 640), dtype=np.uint8)
    # The windshield ends around two thirds down these frames.
    # Use a central trapezoid above the dashboard, widening toward the road.
    road = np.int32(regions['road_polygon'])
    if upper_scene:
        x1, y1, x2, y2 = regions['upper_rectangle']
        mask[y1:y2, x1:x2] = 255
        if mask_sky:
            hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)
            blue = cv2.inRange(hsv, (82, 18, 105), (132, 255, 255))
            bright_neutral = cv2.inRange(hsv, (0, 0, 190), (179, 42, 255))
            sky = cv2.dilate(blue | bright_neutral, np.ones((7, 7), np.uint8))
            sky = cv2.resize(sky, (640, 480), interpolation=cv2.INTER_NEAREST)
            mask[sky != 0] = 0
    else:
        cv2.fillConvexPoly(mask, road, 255)
    for x1, y1, x2, y2 in excluded_boxes:
        padding = regions['vehicle_padding']
        left, top = max(0, math.floor(x1)-padding), max(0, math.floor(y1)-padding)
        right, bottom = min(640, math.ceil(x2)+padding), min(480, math.ceil(y2)+padding)
        mask[top:bottom, left:right] = 0
    if scale != 1:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    points, descriptors = detector.detectAndCompute(gray, mask)
    if scale != 1:
        for point in points:
            x, y = point.pt
            point.pt = (x/scale, y/scale)
    return points, descriptors


def motion(a, b, matcher, settings, regions, upper_scene=False):
    if a is None or b is None or a[1] is None or b[1] is None:
        return {'reason': 'no usable road features'}
    pairs = matcher.knnMatch(a[1], b[1], k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < settings['match_ratio']*n.distance]
    unique = {}
    for match in sorted(good, key=lambda m: m.distance):
        unique.setdefault(match.trainIdx, match)
    if len(unique) < settings['min_matches']:
        return {'matches': len(unique), 'reason': 'too few distinct matches'}
    points = np.float32([a[0][m.queryIdx].pt for m in unique.values()])
    target = np.float32([b[0][m.trainIdx].pt for m in unique.values()])
    transform, mask = cv2.findHomography(points, target, cv2.RANSAC, settings['ransac_pixels'])
    if transform is None or mask is None:
        return {'matches': len(unique), 'reason': 'alignment failed'}
    inliers = mask.ravel().astype(bool)
    n = int(inliers.sum())
    spread = np.ptp(points[inliers], axis=0) if n else [0, 0]
    quality = {'matches': len(points), 'inliers': n, 'inlier_ratio': round(n/len(points), 3),
               'spread_x': round(float(spread[0]), 2), 'spread_y': round(float(spread[1]), 2)}
    if n < settings['min_inliers'] or n/len(points) < settings['min_inlier_ratio'] or np.any(np.asarray(spread) < settings['min_spread_pixels']):
        return {**quality, 'reason': 'weak or localized alignment'}
    rows = regions['upper_grid_y'] if upper_scene else regions['road_grid_y']
    grid = np.float32([[x, y] for y in rows for x in regions['grid_x']])
    shift = cv2.perspectiveTransform(grid[None], transform)[0] - grid
    if not np.isfinite(shift).all():
        return {**quality, 'reason': 'unstable transform'}
    vertical = float(np.median(np.abs(shift[:, 1])) / 480)
    horizontal = float(np.max(np.abs(shift[:, 0])) / 640)
    peak = float(np.max(np.linalg.norm(shift, axis=1)) / 480)
    return {**quality, 'score': round(vertical, 6), 'horizontal': round(horizontal, 6),
            'peak': round(peak, 6)}


