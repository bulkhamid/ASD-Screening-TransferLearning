import cv2
import numpy as np


def resize_with_letterbox(frame, target_size, pad_value=128, return_mask=False):
    """Resize without distorting aspect ratio, then center-pad to target size."""
    target_w, target_h = target_size
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
        mask = np.zeros((target_h, target_w), dtype=bool)
        return (canvas, mask) if return_mask else canvas

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), pad_value, dtype=resized.dtype)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    if not return_mask:
        return canvas
    mask = np.zeros((target_h, target_w), dtype=bool)
    mask[y0:y0 + new_h, x0:x0 + new_w] = True
    return canvas, mask


def normalize_imagenet(video):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None, None]
    return (video - mean) / std
