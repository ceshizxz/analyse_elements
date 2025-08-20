import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte
from pathlib import Path

input_dir = r"E:\zxz\project\extracted\b\cropped"
output_dir = r"E:\zxz\project\extracted\b\preprocessed"

target_size = 224         # final image size (square canvas)
min_long_side = 300       # minimum long side before resize
sauvola_window = 31       # window size for Sauvola threshold
sauvola_k = 0.2           # k parameter for Sauvola threshold
open_kernel = 3           # kernel size for opening
close_kernel = 3          # kernel size for closing
open_iter = 1             # number of opening iterations
close_iter = 1            # number of closing iterations
hough_thresh = 80         # threshold for Hough line transform
canny_low = 50            # Canny low threshold
canny_high = 150          # Canny high threshold

def ensure_dir(p: Path):
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)

def to_gray(img):
    """Convert to grayscale if image is RGB."""
    return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def sauvola_binarize(gray):
    """Adaptive binarisation using Sauvola thresholding."""
    g = gray.astype(np.float32) / 255.0
    thresh = threshold_sauvola(g, window_size=sauvola_window, k=sauvola_k)
    bin_img = (g > thresh).astype(np.uint8) * 255
    # Invert if mostly white foreground
    if bin_img.mean() > 127:
        bin_img = 255 - bin_img
    return bin_img

def morphology_clean(bin_img):
    """Noise removal using opening + closing morphology."""
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
    out = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, open_k, iterations=open_iter)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, close_k, iterations=close_iter)
    return out

def estimate_skew_angle(bin_img):
    """Estimate skew angle using Hough transform of edges."""
    edges = cv2.Canny(bin_img, canny_low, canny_high, L2gradient=True)
    lines = cv2.HoughLines(edges, 1, np.pi/180, hough_thresh)
    if lines is None:
        return 0.0
    angles = []
    for rho, theta in lines[:,0,:]:
        angle = (theta * 180.0 / np.pi) - 90.0
        if -85 < angle < 85:  # exclude near vertical/horizontal lines
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0

def rotate_image(img, angle_deg):
    """Rotate image around its center and expand canvas."""
    h, w = img.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = np.abs(M[0,0]); sin = np.abs(M[0,1])
    # new size after rotation
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0,2] += (new_w/2) - center[0]
    M[1,2] += (new_h/2) - center[1]
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)

def resize_with_padding(img):
    """Resize image keeping aspect ratio, pad to target_size."""
    h, w = img.shape[:2]
    long_side = max(h, w)
    # Scale up if too small
    if long_side < min_long_side:
        scale_up = min_long_side / long_side
        img = cv2.resize(img, (int(w*scale_up), int(h*scale_up)), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
    # Scale down to fit target_size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    # Center on white canvas
    canvas = np.full((target_size, target_size), 255, dtype=resized.dtype)
    y0, x0 = (target_size - new_h)//2, (target_size - new_w)//2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

def to_edge(img_gray):
    """Extract edges using Canny operator."""
    return cv2.Canny(img_gray, canny_low, canny_high, L2gradient=True)

def to_skeleton(bin_img):
    """Convert binary image to skeleton representation."""
    skel = skeletonize((bin_img > 127).astype(bool))
    return img_as_ubyte(skel)

def process_one(path, out_dir):
    """Full preprocessing pipeline for one image."""
    stem = Path(path).stem
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Cannot read: {path}")
        return

    # Convert to grayscale
    gray = to_gray(img)
    cv2.imwrite(str(out_dir / f"{stem}_gray.png"), gray)

    # Sauvola binarisation
    bin_img = sauvola_binarize(gray)
    cv2.imwrite(str(out_dir / f"{stem}_bin.png"), bin_img)

    # Morphological noise cleaning
    den = morphology_clean(bin_img)
    cv2.imwrite(str(out_dir / f"{stem}_denoise.png"), den)

    # Deskew using Hough-based angle estimation
    angle = estimate_skew_angle(den)
    deskew = rotate_image(den, -angle)
    cv2.imwrite(str(out_dir / f"{stem}_deskew.png"), deskew)

    # Resize + pad to fixed canvas
    norm = resize_with_padding(deskew)
    cv2.imwrite(str(out_dir / f"{stem}_norm.png"), norm)

    # Edge map
    edge = to_edge(norm)
    cv2.imwrite(str(out_dir / f"{stem}_edge.png"), edge)

    # Skeletonisation
    skel = to_skeleton(norm)
    cv2.imwrite(str(out_dir / f"{stem}_skeleton.png"), skel)

    print(f"[DONE] {path} -> angle={angle:.2f}°")

if __name__ == "__main__":
    # Prepare directories
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    ensure_dir(out_dir)

    images = [p for p in in_dir.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]]
    print(f"[INFO] Found {len(images)} images.")

    for p in images:
        process_one(p, out_dir)

    print("[INFO] All done.")
