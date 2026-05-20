import cv2
import numpy as np

def bgr_to_ycbcr(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert BGR frame to YCbCr and apply 4:2:0 chroma subsampling.
    Returns Y (full-res), Cb (half-res), Cr (half-res).
    """
    # OpenCV converts BGR to YCrCb (order is Y, Cr, Cb)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    Y = ycrcb[:, :, 0]
    Cr = ycrcb[:, :, 1]
    Cb = ycrcb[:, :, 2]
    
    # 4:2:0 Chroma subsampling: Downsample Cb and Cr by a factor of 2 in both dimensions
    height, width = frame.shape[:2]
    
    # Using INTER_AREA for downsampling
    Cb_sub = cv2.resize(Cb, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    Cr_sub = cv2.resize(Cr, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    
    return Y, Cb_sub, Cr_sub

def ycbcr_to_bgr(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """
    Upsample Cb/Cr back to full-res, merge, convert to BGR.
    """
    height, width = Y.shape[:2]
    
    # Upsample Cb and Cr back to full resolution
    # Using INTER_LINEAR for upsampling
    Cb_up = cv2.resize(Cb, (width, height), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Merge channels. OpenCV expects YCrCb (Y, Cr, Cb)
    ycrcb = np.stack([Y, Cr_up, Cb_up], axis=2)
    
    # Convert back to BGR
    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return bgr
