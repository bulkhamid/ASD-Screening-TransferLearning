"""
Utility functions for video frame processing with aspect ratio preservation.
"""
import cv2
import numpy as np


def resize_keep_aspect_ratio(frame, target_size=(112, 112), pad_color=(0, 0, 0)):
    """
    Resize frame while preserving aspect ratio, using padding to reach target size.
    
    For videos originally 1280x720, this function will resize to fit within the 
    target dimensions while maintaining the 16:9 aspect ratio, then pad the 
    remaining space with black pixels.
    
    Args:
        frame: Input frame as numpy array (H, W, C)
        target_size: Target size as (height, width) tuple
        pad_color: Color to use for padding (default: black)
    
    Returns:
        Resized and padded frame of shape (target_size[0], target_size[1], C)
    """
    target_h, target_w = target_size
    h, w = frame.shape[:2]
    
    # Calculate scaling factor to fit within target size
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize the frame
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded frame
    if len(frame.shape) == 3:
        padded = np.full((target_h, target_w, frame.shape[2]), pad_color, dtype=frame.dtype)
    else:
        padded = np.full((target_h, target_w), pad_color[0], dtype=frame.dtype)
    
    # Calculate padding offsets to center the resized frame
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    # Place resized frame in center of padded frame
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded
