"""YOLO box comparison for preserving materially changed labels."""
def read_annotations(path):
    boxes = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            category, x, y, w, h = line.split()
            x, y, w, h = map(float, (x, y, w, h))
            boxes.append((category, x-w/2, y-h/2, x+w/2, y+h/2))
    return boxes


def annotations_match(left, right):
    if len(left) != len(right):
        return False
    available = list(right)
    for category, x1, y1, x2, y2 in left:
        area = (x2-x1)*(y2-y1)
        best = None
        best_iou = .65
        for index, (other, a, b, c, d) in enumerate(available):
            other_area = (c-a)*(d-b)
            if category != other or min(area, other_area) <= 0 or min(area, other_area)/max(area, other_area) < .75:
                continue
            overlap = max(0, min(x2,c)-max(x1,a))*max(0, min(y2,d)-max(y1,b))
            iou = overlap/(area+other_area-overlap)
            if iou >= best_iou:
                best, best_iou = index, iou
        if best is None:
            return False
        available.pop(best)
    return True


