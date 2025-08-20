from pathlib import Path
import cv2
import numpy as np
import pandas as pd

results_csv       = r"E:\zxz\project\extracted\b\stageB_results.csv"
candidates_csv    = r"E:\zxz\project\extracted\b\stageA_candidates.csv"
preprocessed_dir  = Path(r"E:\zxz\project\extracted\b\preprocessed")   # Folder of preprocessed images
overlay_dir       = Path(r"E:\zxz\project\extracted\b\overlay")       # Output folder for overlay triplets

panel_side        = 224
canny_sigma       = 0.33   # Sigma for auto Canny edge detection

# Overlay colors
COLOR_LEFT  = (0, 255, 0)
COLOR_RIGHT = (180, 105, 255)
COLOR_BOTH  = (150, 0, 150)

label_band_px  = 28
text_scale     = 0.6
text_thickness = 2

# ==============================
# Utility Functions
# ==============================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def imread_gray(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

def letterbox_pad(gray: np.ndarray, target_h: int, target_w: int, pad_value: int = 255):
    h, w = gray.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((target_h, target_w), pad_value, dtype=np.uint8)
    dx = (target_w - new_w) // 2
    dy = (target_h - new_h) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized
    return canvas

def put_centered_text(img, text, y, color=(0,0,0), scale=0.6, thickness=2):
    """Draw centered text at a given vertical position."""
    if not isinstance(text, str) or text.strip() == "":
        return
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (img.shape[1] - tw) // 2
    y2 = max(y, th + baseline + 2)
    cv2.putText(img, text, (x, y2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def auto_canny(img_gray, sigma=0.33):
    """Perform automatic Canny edge detection using median-based thresholds."""
    v = float(np.median(img_gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if lower >= upper:
        lower = max(0, upper - 1)
    return cv2.Canny(img_gray, lower, upper, L2gradient=True)

def overlay_edges_bgr_from_letterboxed(A_gray: np.ndarray, B_gray: np.ndarray) -> np.ndarray:
    """
    Overlay Canny edges from two images (A and B).
    Different colors represent unique or shared edges.
    """
    if A_gray.shape != B_gray.shape:
        raise ValueError("overlay input size mismatch")
    A_edge = auto_canny(A_gray, sigma=canny_sigma)
    B_edge = auto_canny(B_gray, sigma=canny_sigma)
    h, w = A_edge.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    maskA = A_edge > 0
    maskB = B_edge > 0
    both  = maskA & maskB
    canvas[np.logical_and(maskA, np.logical_not(both))] = COLOR_LEFT
    canvas[np.logical_and(maskB, np.logical_not(both))] = COLOR_RIGHT
    canvas[both] = COLOR_BOTH
    return canvas


# ==============================
# Panel Construction
# ==============================
def build_triplet_keep_aspect(A_gray: np.ndarray, B_gray: np.ndarray,
                              labelA: str, labelB: str, sim_text: str | None = None,
                              side: int = 224) -> np.ndarray:
    A_pad = letterbox_pad(A_gray, side, side, pad_value=255)
    B_pad = letterbox_pad(B_gray, side, side, pad_value=255)
    overlay = overlay_edges_bgr_from_letterboxed(A_pad, B_pad)
    A_bgr = cv2.cvtColor(A_pad, cv2.COLOR_GRAY2BGR)
    B_bgr = cv2.cvtColor(B_pad, cv2.COLOR_GRAY2BGR)

    # Top bands with text labels
    bandA = np.full((label_band_px, side, 3), 255, dtype=np.uint8)
    bandB = np.full((label_band_px, side, 3), 255, dtype=np.uint8)
    bandO = np.full((label_band_px, side, 3), 255, dtype=np.uint8)
    put_centered_text(bandA, labelA, int(label_band_px*0.65), scale=text_scale, thickness=text_thickness)
    put_centered_text(bandB, labelB, int(label_band_px*0.65), scale=text_scale, thickness=text_thickness)
    put_centered_text(bandO, ("Overlay  |  " + sim_text) if isinstance(sim_text, str) else "Overlay",
                      int(label_band_px*0.65), scale=text_scale, thickness=text_thickness)

    # Stack vertically (label + image) and then horizontally
    A_panel = np.vstack([bandA, A_bgr])
    B_panel = np.vstack([bandB, B_bgr])
    O_panel = np.vstack([bandO, overlay])
    return np.concatenate([A_panel, B_panel, O_panel], axis=1)


# ==============================
# Path & CSV Utilities
# ==============================
def path_to_id_like(s: str) -> str:
    """Convert file path to an ID-like string (removing '_norm' if present)."""
    p = Path(s)
    name = p.stem
    if name.endswith("_norm"):
        name = name[:-5]
    return name

def id_to_preprocessed_norm(id_str: str) -> Path:
    """Build the normalized preprocessed path for a given ID."""
    return preprocessed_dir / f"{id_str}_norm.png"

def find_companion(path: Path, suffix: str):
    """Find a related file with a different suffix (e.g., '_norm')."""
    s = path.name
    if s.endswith("_norm.png"):
        return path.with_name(s.replace("_norm.png", f"{suffix}.png"))
    else:
        stem = path.stem
        return path.with_name(f"{stem}{suffix}.png")

def load_candidates_index(csv_path: Path):
    """
    Load candidate images from CSV and map their IDs to file paths.
    This helps resolve paths when using non-preprocessed inputs.
    """
    dfA = pd.read_csv(csv_path)
    mapping = {}
    for col in ["imgA", "imgB"]:
        for raw in dfA[col].tolist():
            rp = Path(str(raw).replace("\\", "/"))
            key = path_to_id_like(rp.name)
            mapping.setdefault(key, rp)
    return mapping

def main():
    ensure_dir(overlay_dir)

    df = pd.read_csv(results_csv)
    assert {"imgA","imgB","sim_cosine","decision"}.issubset(df.columns), \
        "results_csv must have columns: imgA, imgB, sim_cosine, decision"

    # Decide whether to use preprocessed images
    use_preprocessed = preprocessed_dir.exists()
    id2path = {} if use_preprocessed else load_candidates_index(Path(candidates_csv))

    seen_pairs = set()
    saved = 0

    for idx, row in df.iterrows():
        if int(row["decision"]) != 1:   # Only process positive matches
            continue

        idA = path_to_id_like(str(row["imgA"]))
        idB = path_to_id_like(str(row["imgB"]))
        sim  = float(row["sim_cosine"])

        key = tuple(sorted([idA, idB]))
        if key in seen_pairs:   # Skip duplicates
            continue
        seen_pairs.add(key)

        # Get file paths
        if use_preprocessed:
            pA_norm = id_to_preprocessed_norm(idA)
            pB_norm = id_to_preprocessed_norm(idB)
        else:
            baseA = id2path.get(idA, Path(idA))
            baseB = id2path.get(idB, Path(idB))
            pA_norm = find_companion(Path(str(baseA).replace("\\", "/")), "_norm")
            pB_norm = find_companion(Path(str(baseB).replace("\\", "/")), "_norm")

        try:
            A_gray = imread_gray(pA_norm)
            B_gray = imread_gray(pB_norm)
            if A_gray is None or B_gray is None:
                raise FileNotFoundError(f"Cannot read: {pA_norm} or {pB_norm}")

            # Build triplet visualization panel
            trip = build_triplet_keep_aspect(A_gray, B_gray, idA, idB, f"sim={sim:.3f}",
                                             side=panel_side)
            out_name = f"{idx:06d}_{sim:.3f}_{idA}__{idB}_triplet_keep_ar.png"
            cv2.imwrite(str(overlay_dir / out_name), trip)
            saved += 1
        except Exception as e:
            print(f"[WARN] overlay failed for row {idx}: {e}")

    print(f"[INFO] Triplets saved -> {overlay_dir} (count={saved})")
    if use_preprocessed:
        print(f"[INFO] Read from preprocessed: {preprocessed_dir}")
    else:
        print(f"[INFO] Rebuilt paths using: {candidates_csv}")

if __name__ == "__main__":
    main()
