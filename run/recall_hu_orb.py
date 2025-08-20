import cv2
import numpy as np
import pandas as pd
from pathlib import Path

input_dir   = r"E:\zxz\project\extracted\b\preprocessed"
output_csv  = r"E:\zxz\project\extracted\b\stageA_candidates.csv"
use_only_norm = True
K_recall      = 200 
K_save_each   = 50
max_images    = None

w_orb      = 0.2
w_chamfer  = 0.5
w_hu       = 0.3
hu_eps     = 1e-6

# Edge / Skeleton
canny_sigma = 0.33
hu_on_skeleton = True

ecc_iters   = 1000
ecc_eps     = 1e-5
use_ecc     = True
ransac_min_matches = 8

try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

# ------------- IO -------------
def list_images(folder: Path, only_norm=True):
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
    if only_norm:
        return sorted([p for p in folder.glob("*_norm.png") if p.is_file()])
    return sorted([p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in exts])

def imread_gray(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {p}")
    return img

# --------- preprocess utils ----------
def binarize_otsu(gray):
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bin_img.mean() > 127:
        bin_img = 255 - bin_img
    return bin_img

def to_skeleton(bin_img):
    try:
        from skimage.morphology import skeletonize
        from skimage.util import img_as_ubyte
        skel = skeletonize((bin_img < 128).astype(bool))
        return img_as_ubyte(skel)
    except Exception:
        bw = cv2.threshold(bin_img, 127, 255, cv2.THRESH_BINARY_INV)[1]
        skel = np.zeros_like(bw)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        done = False
        temp = np.zeros_like(bw)
        eroded = np.zeros_like(bw)
        while not done:
            cv2.erode(bw, element, eroded)
            cv2.dilate(eroded, element, temp)
            cv2.subtract(bw, temp, temp)
            cv2.bitwise_or(skel, temp, skel)
            bw = eroded.copy()
            if cv2.countNonZero(bw) == 0:
                done = True
        return skel

def huextract(gray, use_skeleton=True):
    bin_img = binarize_otsu(gray)
    src = to_skeleton(bin_img) if use_skeleton else bin_img
    m = cv2.moments((src < 128).astype(np.uint8), binaryImage=True)
    hu = cv2.HuMoments(m).flatten()
    with np.errstate(divide='ignore'):
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu.astype(np.float32)

def standardize_features(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-9
    return (X - mu) / sigma, mu, sigma

def build_index(Xz):
    if HAVE_FAISS:
        idx = faiss.IndexFlatL2(Xz.shape[1])
        idx.add(Xz.astype(np.float32))
        return idx
    return None

def search_topk(index, Xz, qvec, K):
    if HAVE_FAISS and index is not None:
        D, I = index.search(qvec[None, :].astype(np.float32), K+1)
        return D[0], I[0]
    diffs = Xz - qvec[None, :]
    dists = np.sum(diffs*diffs, axis=1)
    idxs = np.argsort(dists)[:K+1]
    return dists[idxs], idxs

def hu_similarity(d2):
    return 1.0 / (1.0 + d2 + hu_eps)

# -------- edges & canny --------
def auto_canny(gray, sigma=0.33):
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if lower >= upper:
        lower = max(0, upper - 1)
    return cv2.Canny(gray, lower, upper, L2gradient=True)

# ---------- alignment ----------
def ecc_align_affine(src_gray, dst_gray, iters=1000, eps=1e-5):
    """
    Align dst to src using ECC affine; return warp matrix 2x3 and warped image.
    """
    try:
        src = src_gray.astype(np.float32) / 255.0
        dst = dst_gray.astype(np.float32) / 255.0
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
        _, warp = cv2.findTransformECC(src, dst, warp, cv2.MOTION_AFFINE,
                                       criteria, inputMask=None, gaussFiltSize=5)
        h, w = src_gray.shape[:2]
        aligned = cv2.warpAffine(dst_gray, warp, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)
        return True, warp, aligned
    except Exception:
        return False, None, None

def ransac_homography_align(edgeA, edgeB, min_matches=8):
    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=5, scaleFactor=1.2,
                         nlevels=8, edgeThreshold=15)
    kpsA, desA = orb.detectAndCompute(edgeA, None)
    kpsB, desB = orb.detectAndCompute(edgeB, None)
    if desA is None or desB is None or len(kpsA) == 0 or len(kpsB) == 0:
        return False, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desA, desB, k=2)

    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < min_matches:
        return False, None, None

    ptsA = np.float32([kpsA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 3.0)  # warp B->A
    if H is None:
        return False, None, None

    h, w = edgeA.shape[:2]
    warped = cv2.warpPerspective(edgeB, H, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    return True, H, warped

# ---------- scoring ----------
def orb_score(edgeA, edgeB):
    if edgeA is None or edgeB is None:
        return 0.0
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=5, scaleFactor=1.2,
                         nlevels=8, edgeThreshold=15)
    kpsA, desA = orb.detectAndCompute(edgeA, None)
    kpsB, desB = orb.detectAndCompute(edgeB, None)
    if desA is None or desB is None or len(kpsA) == 0 or len(kpsB) == 0:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desA, desB, k=2)

    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    inliers = 0
    if len(good) >= 8:
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 3.0)
        if mask is not None:
            inliers = int(mask.sum())

    denom = max(1, min(len(kpsA), len(kpsB)))
    score = (inliers if inliers > 0 else len(good)) / denom
    return float(np.clip(score, 0.0, 1.0))

def chamfer_similarity(edgeA, edgeB):
    """
    Smaller chamfer distance -> higher similarity.
    Convert to [0,1] similarity by exp(-d / scale).
    """
    if edgeA is None or edgeB is None:
        return 0.0
    A = (edgeA > 0).astype(np.uint8)
    B = (edgeB > 0).astype(np.uint8)
    if A.sum() == 0 or B.sum() == 0:
        return 0.0
    # distance transform on complement (so DT=0 on edges)
    dt = cv2.distanceTransform(255 - edgeA, cv2.DIST_L2, 3).astype(np.float32)
    ys, xs = np.where(B > 0)
    if len(xs) == 0:
        return 0.0
    d = float(dt[ys, xs].mean())
    # scale by image size to be roughly invariant
    h, w = edgeA.shape[:2]
    scale = 0.02 * (h + w)  # heuristic
    sim = float(np.exp(-d / max(1e-6, scale)))
    return float(np.clip(sim, 0.0, 1.0))

def ssim_edges(edgeA, edgeB):
    try:
        from skimage.metrics import structural_similarity as ssim
        if edgeA is None or edgeB is None:
            return 0.0
        a = (edgeA.astype(np.float32))/255.0
        b = (edgeB.astype(np.float32))/255.0
        s, _ = ssim(a, b, full=True, data_range=1.0)
        return float(np.clip(s, 0.0, 1.0))
    except Exception:
        return 0.0

# ---------- overlay ----------
def save_overlay_rgb(edgeA, edgeB, out_path):
    if edgeA is None or edgeB is None:
        return
    h, w = edgeA.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 1] = edgeA  # green
    rgb[..., 2] = edgeB  # red
    cv2.imwrite(str(out_path), rgb)

# ---------------- main ----------------
def main():
    in_dir = Path(input_dir)
    paths = list_images(in_dir, only_norm=use_only_norm)
    if max_images is not None:
        paths = paths[:max_images]
    assert len(paths) >= 2, "Need at least 2 images."

    print(f"[INFO] Loading {len(paths)} images from {in_dir}")

    # load
    grays = [imread_gray(p) for p in paths]
    # recall features: skeleton-Hu
    hu_list = [huextract(g, use_skeleton=hu_on_skeleton) for g in grays]
    X = np.vstack(hu_list).astype(np.float32)
    Xz, mu, sigma = standardize_features(X)

    # index
    index = build_index(Xz)
    if HAVE_FAISS and index is not None:
        print("[INFO] Using FAISS IndexFlatL2 for recall.")
    else:
        print("[WARN] FAISS not available -> brute-force recall.")

    # edges
    edges = [auto_canny(g, sigma=canny_sigma) for g in grays]

    N = len(paths)
    rows = []

    for qi, (qpath, qvec) in enumerate(zip(paths, Xz), start=1):
        D, I = search_topk(index, Xz, qvec, K_recall)
        cand_idxs = [int(idx) for idx in I if int(idx) != (qi-1)]
        cand_dists = [float(d) for idx, d in zip(I, D) if int(idx) != (qi-1)]

        q_gray = grays[qi-1]
        q_edge = edges[qi-1]

        for idx, d2 in zip(cand_idxs, cand_dists):
            c_gray = grays[idx]
            c_edge = edges[idx]

            # --- alignment: ECC affine (gray) ---
            aligned_edge = None
            ok = False
            if use_ecc:
                ok, warp, aligned_gray = ecc_align_affine(q_gray, c_gray, iters=ecc_iters, eps=ecc_eps)
                if ok and aligned_gray is not None:
                    aligned_edge = auto_canny(aligned_gray, sigma=canny_sigma)

            # fallback to RANSAC homography on edges if ECC failed
            if not ok or aligned_edge is None:
                ok2, H, warped = ransac_homography_align(q_edge, c_edge, min_matches=ransac_min_matches)
                if ok2 and warped is not None:
                    aligned_edge = warped
                else:
                    aligned_edge = c_edge  # no alignment

            # --- scores on aligned edges ---
            orb = orb_score(q_edge, aligned_edge)
            cham = chamfer_similarity(q_edge, aligned_edge)
            hu_sim = hu_similarity(d2)
            ssim_val = ssim_edges(q_edge, aligned_edge)
            cham = 0.85 * cham + 0.15 * ssim_val

            fused = w_orb * orb + w_chamfer * cham + w_hu * hu_sim

            rows.append({
                "imgA": str(paths[qi-1]),
                "imgB": str(paths[idx]),
                "hu_dist2": d2,
                "hu_sim": hu_sim,
                "orb_score": orb,
                "chamfer_sim": cham,
                "ssim_edge": ssim_val,
                "score_fused": fused
            })

        print(f"[{qi}/{N}] {qpath.name} -> candidates: {len(cand_idxs)}")

    df = pd.DataFrame(rows)
    if K_save_each is not None:
        df = (df.sort_values(["imgA","score_fused"], ascending=[True, False])
                .groupby("imgA", as_index=False)
                .head(K_save_each))
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved candidates to: {out}")

if __name__ == "__main__":
    main()
