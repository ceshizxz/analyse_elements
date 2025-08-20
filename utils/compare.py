from pathlib import Path
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Input/Output settings
candidates_csv = r"E:\zxz\project\extracted\b\stageA_candidates.csv"
output_csv     = r"E:\zxz\project\extracted\b\stageB_results.csv"
tau_similarity = 0.95   # Threshold for cosine similarity

# Image preprocessing settings
expected_size  = 224
use_pretrained = True
normalize_type = "imagenet"

# Directory for saving montage comparison images
montage_dir    = r"E:\zxz\project\extracted\b\compare"

# ---------- Utility functions ----------

def imread_gray(path: Path):
    """Read image as grayscale"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img

def ensure_dir(p: Path):
    """Create directory if not exists"""
    p.mkdir(parents=True, exist_ok=True)

def safe_path(p):
    """Convert path to safe format (avoid backslashes)"""
    return Path(str(p).replace("\\", "/"))

def find_companion(path: Path, suffix: str):
    """Find companion image file with a given suffix"""
    s = path.name
    if s.endswith("_norm.png"):
        return path.with_name(s.replace("_norm.png", f"{suffix}.png"))
    else:
        stem = path.stem
        return path.with_name(f"{stem}{suffix}.png")

def auto_canny(img_gray):
    """Apply Canny edge detection with fixed thresholds"""
    return cv2.Canny(img_gray, 50, 150, L2gradient=True)

def auto_skeleton(img_norm):
    """Compute skeleton of a binary image.
       If skimage is available, use skeletonize; otherwise fallback to OpenCV iterative thinning."""
    try:
        from skimage.morphology import skeletonize
        from skimage.util import img_as_ubyte
        _, bin_img = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if bin_img.mean() > 127:
            bin_img = 255 - bin_img
        skel = skeletonize((bin_img > 127).astype(bool))
        skel_u8 = img_as_ubyte(skel)
        return skel_u8
    except Exception:
        bin_img = cv2.adaptiveThreshold(img_norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, 21, 5)
        skel = np.zeros_like(bin_img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        done = False
        temp = np.zeros_like(bin_img)
        eroded = np.zeros_like(bin_img)
        while not done:
            cv2.erode(bin_img, element, eroded)
            cv2.dilate(eroded, element, temp)
            cv2.subtract(bin_img, temp, temp)
            cv2.bitwise_or(skel, temp, skel)
            bin_img = eroded.copy()
            if cv2.countNonZero(bin_img) == 0:
                done = True
        return skel

def load_three_channel(norm_path: Path):
    """Load norm, edge, and skeleton images as a 3-channel input"""
    norm = imread_gray(norm_path)
    if norm is None:
        raise FileNotFoundError(f"Cannot read: {norm_path}")

    edge_path = find_companion(norm_path, "_edge")
    skel_path = find_companion(norm_path, "_skeleton")

    if edge_path.exists():
        edge = imread_gray(edge_path)
    else:
        edge = auto_canny(norm)

    if skel_path.exists():
        skel = imread_gray(skel_path)
    else:
        skel = auto_skeleton(norm)

    def ensure_size(img, h, w):
        """Resize image to target size if needed"""
        if img.shape[0] != h or img.shape[1] != w:
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        return img

    H, W = norm.shape[:2]
    edge = ensure_size(edge, H, W)
    skel = ensure_size(skel, H, W)

    ch3 = np.stack([norm, edge, skel], axis=2)  # H, W, 3
    return ch3

def to_tensor(ch3: np.ndarray):
    """Convert numpy image (HWC) to PyTorch tensor (1x3xH xW) with optional normalization"""
    x = ch3.astype(np.float32) / 255.0
    x = np.transpose(x, (2,0,1))  # 3xHxW
    x = torch.from_numpy(x)[None, ...]  # 1x3xHxW
    if normalize_type == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        x = (x - mean) / std
    return x

# ---------- Neural Network Model ----------

class ResNet18Embed(nn.Module):
    """ResNet18 backbone for embedding extraction (L2 normalized features)"""
    def __init__(self, pretrained=True, out_dim=512):
        super().__init__()
        import torchvision.models as models
        if pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                backbone = models.resnet18(weights=weights)
            except Exception:
                backbone = models.resnet18(weights=None)
        else:
            backbone = models.resnet18(weights=None)
        modules = list(backbone.children())[:-1]  # remove final FC layer
        self.feature = nn.Sequential(*modules)
        self.out_dim = out_dim

    def forward(self, x):
        feats = self.feature(x)
        feats = feats.view(feats.size(0), -1)
        feats = F.normalize(feats, p=2, dim=1)  # L2 normalize embeddings
        return feats

def cosine_sim(e1: torch.Tensor, e2: torch.Tensor):
    """Compute cosine similarity between two embeddings"""
    return F.cosine_similarity(e1, e2, dim=1)

# ---------- Visualization ----------

def build_montage(imgA_path: Path, imgB_path: Path, out_path: Path, text=""):
    """Concatenate two images side by side with optional similarity text"""
    try:
        A = cv2.imread(str(imgA_path), cv2.IMREAD_GRAYSCALE)
        B = cv2.imread(str(imgB_path), cv2.IMREAD_GRAYSCALE)
        if A is None or B is None:
            return
        H, W = expected_size, expected_size
        A = cv2.resize(A, (W,H), interpolation=cv2.INTER_NEAREST)
        B = cv2.resize(B, (W,H), interpolation=cv2.INTER_NEAREST)
        C = np.concatenate([A, B], axis=1)
        C = cv2.cvtColor(C, cv2.COLOR_GRAY2BGR)
        if text:
            cv2.putText(C, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_path), C)
    except Exception:
        pass

def path_to_id(p: Path):
    """Convert image path to ID (strip '_norm' suffix if present)"""
    name = Path(p).stem
    if name.endswith("_norm"):
        name = name[:-5]
    return name

# ---------- Main Pipeline ----------

def main():
    # Load candidate CSV
    cand_path = Path(candidates_csv)
    assert cand_path.exists(), f"CSV not found: {cand_path}"
    df = pd.read_csv(cand_path)
    if not {"imgA","imgB"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: imgA, imgB")

    # Collect all image paths for embedding extraction
    all_paths = set()
    for col in ["imgA", "imgB"]:
        for p in df[col].tolist():
            p = safe_path(p)
            p = Path(p)
            if not p.name.endswith("_norm.png"):
                norm_p = find_companion(p, "_norm")
                if norm_p.exists():
                    p = norm_p
            all_paths.add(str(p))
    all_paths = [Path(p) for p in sorted(all_paths)]

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Embed(pretrained=use_pretrained).to(device)
    model.eval()

    # Build embeddings cache
    cache = {}
    with torch.no_grad():
        for i, p in enumerate(all_paths, 1):
            ch3 = load_three_channel(p)
            if ch3.shape[0] != expected_size or ch3.shape[1] != expected_size:
                ch3 = cv2.resize(ch3, (expected_size, expected_size), interpolation=cv2.INTER_NEAREST)
            x = to_tensor(ch3).to(device)
            emb = model(x)
            cache[str(p)] = emb.detach().cpu()
            if i % 100 == 0:
                print(f"[Emb] {i}/{len(all_paths)}")

    sims = []
    decisions = []

    ensure_dir(Path(montage_dir))
    saved = 0
    seen_pairs = set()  # Track unordered pairs to avoid duplicate montages

    # Pairwise comparison
    for idx, row in df.iterrows():
        pA = safe_path(row["imgA"])
        pB = safe_path(row["imgB"])

        # Ensure normalized image paths
        pA_norm = Path(pA)
        if not pA_norm.name.endswith("_norm.png"):
            pA_norm = find_companion(pA_norm, "_norm")
        pB_norm = Path(pB)
        if not pB_norm.name.endswith("_norm.png"):
            pB_norm = find_companion(pB_norm, "_norm")

        # Compute embeddings if missing
        if str(pA_norm) not in cache or str(pB_norm) not in cache:
            for pN in [pA_norm, pB_norm]:
                if str(pN) not in cache:
                    ch3 = load_three_channel(pN)
                    if ch3.shape[0] != expected_size or ch3.shape[1] != expected_size:
                        ch3 = cv2.resize(ch3, (expected_size, expected_size), interpolation=cv2.INTER_NEAREST)
                    x = to_tensor(ch3).to(device)
                    with torch.no_grad():
                        cache[str(pN)] = model(x).detach().cpu()

        # Compute cosine similarity
        eA = cache[str(pA_norm)]
        eB = cache[str(pB_norm)]
        sim = float(cosine_sim(eA, eB).item())
        sims.append(sim)
        decisions.append(1 if sim >= tau_similarity else 0)

        # Save montage if above threshold and not duplicated
        if sim >= tau_similarity:
            idA = path_to_id(pA_norm)
            idB = path_to_id(pB_norm)
            key = tuple(sorted([idA, idB]))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            nameA = Path(pA).name
            nameB = Path(pB).name
            out_name = f"{idx:06d}_{sim:.3f}_Y_{nameA}__{nameB}.png"
            out_path = Path(montage_dir) / out_name
            build_montage(pA_norm, pB_norm, out_path, text=f"sim={sim:.3f}")
            saved += 1

    # Save results CSV
    df_out = df.copy()
    df_out["sim_cosine"] = sims
    df_out["decision"] = decisions  # 1: similar, 0: not similar

    if "imgA" in df_out.columns and "imgB" in df_out.columns:
        df_out["imgA"] = df_out["imgA"].apply(lambda x: path_to_id(Path(safe_path(x))))
        df_out["imgB"] = df_out["imgB"].apply(lambda x: path_to_id(Path(safe_path(x))))
    else:
        df_out.iloc[:, 0] = df_out.iloc[:, 0].apply(lambda x: path_to_id(Path(safe_path(x))))
        df_out.iloc[:, 1] = df_out.iloc[:, 1].apply(lambda x: path_to_id(Path(safe_path(x))))

    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved Stage B results -> {output_csv}")
    print(f"[INFO] Similarity threshold tau = {tau_similarity}")

if __name__ == "__main__":
    main()
