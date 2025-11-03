# Lab 4 – Aktyvūs kontūrai: kraštinis (Snakes), regioninis (Chan–Vese) ir geodezinis (MGAC)
# Visi brėžiniai išsaugomi į 'lab4_outputs/'

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize

from skimage import data, img_as_float
from skimage.io import imread
from skimage.filters import gaussian, median
from skimage.morphology import disk, remove_small_objects, binary_closing
from skimage.draw import ellipse as draw_ellipse, disk as draw_disk, polygon as draw_polygon
from skimage.segmentation import (
    active_contour,
    morphological_chan_vese,
    morphological_geodesic_active_contour,
    inverse_gaussian_gradient,
)

from scipy import ndimage

# ---------- Nustatymai (greitesniam veikimui ir aiškiam išvedimui) ----------
MAX_SIZE = 512            # maksimalus ilgesnės kraštinės dydis (pikseliais)
N_ITER_SNAKE = 150        # active_contour iteracijų skaičius
N_ITER_CV = 120           # Chan–Vese iteracijų skaičius
N_ITER_MGAC = 120         # MGAC iteracijų skaičius
RUN_NOISE = False         # triukšmo/inicializacijos analizę išjungiam (buvo lėta)

# failų vardų prefiksai pagal užduoties sekcijas
PFX_S1 = "1_"   # Vaizdų įkėlimas ir peržiūra
PFX_S2 = "2_"   # Kraštų pagrindu: Snakes
PFX_S3 = "3_"   # Regioninis: Chan–Vese
PFX_S4 = "4_"   # Geodezinis: MGAC
PFX_S5 = "5_"   # Palyginimas be GT (metrikos/grafikas)
PFX_A  = "A_"   # Papildoma A) Seed region + Chan–Vese
PFX_B  = "B_"   # Papildoma B) Keli objektai su MGAC

# Heuristiniai MGAC parametrai pagal vaizdo tipą
def pick_mgac_params(name: str):
    n = (name or "").lower()
    if "coin" in n:
        # aiškūs kraštai – šiek tiek daugiau glotninimo ir mažesnis balloon
        return {"alpha": 180.0, "sigma": 1.8, "balloon": 0.45}
    if "cell" in n or "cells3d" in n:
        # neryškūs kraštai, daugiau tekstūros – dar daugiau glotninimo ir kuklesnis balloon
        return {"alpha": 220.0, "sigma": 3.2, "balloon": 0.30}
    # numatytieji
    return {"alpha": 180.0, "sigma": 2.0, "balloon": 0.50}

# ---------- Išsaugojimas ----------
OUT_DIR = Path("lab4_outputs")
OUT_DIR.mkdir(exist_ok=True)

def save_fig(fig, name):
    out_path = OUT_DIR / name
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Išsaugota: {out_path}")

# kelių metodų kontūrai viename paveiksle
import matplotlib.lines as mlines

def save_overlay_multi(img, masks_dict, title, path, color_map=None):
    if color_map is None:
        color_map = {"snakes": "r", "chanvese": "y", "mgac": "c"}
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    handles = []
    for method in ["snakes", "chanvese", "mgac"]:
        if method in masks_dict:
            col = color_map.get(method, 'w')
            plt.contour(masks_dict[method], colors=col, linewidths=1)
            handles.append(mlines.Line2D([], [], color=col, label=method))
    if handles:
        plt.legend(handles=handles, loc='lower right')
    plt.title(title)
    plt.axis('off')
    save_fig(fig, path)

# ---------- Pagalbinės ----------

def to_float_gray(img):
    img = img_as_float(img)
    if img.ndim == 3:
        # jei RGB – paverskime į luminance (paprastas svėrimas)
        img = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    return img

# sumažina vaizdą, jei per didelis (greitesniam veikimui)
def downscale_if_needed(img, max_size=MAX_SIZE):
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_size:
        return img
    scale = max_size / long_edge
    new_h, new_w = int(round(h*scale)), int(round(w*scale))
    return resize(img, (new_h, new_w), anti_aliasing=True, preserve_range=True)


def load_demo_images():
    """Įkelia vietinius 'coins.jpg' ir 'cells.jpg' iš šio failo aplanko. Jei jų nėra, naudoja skimage.data."""
    here = Path(__file__).resolve().parent
    imgs = []
    names = []

    # Vietiniai keliai
    local_files = [(here / "coin.jpg", "coin"), (here / "cell.jpg", "cell")]

    for p, name in local_files:
        if p.exists():
            img = imread(str(p))
            img = to_float_gray(img)
            img = downscale_if_needed(img)
            imgs.append(img)
            names.append(name)
        else:
            print(f"[Įspėjimas] Nerasta {p.name}, naudosiu demo: {name}")
            if name == "coin":
                img = to_float_gray(data.coins())
                img = downscale_if_needed(img)
                imgs.append(img)
                names.append("coin")
            elif name == "cell":
                # bandome sukurti ląstelių projekciją; jei nepavyksta, imame camera
                try:
                    img = to_float_gray(data.cells3d()[:, 1, :, :].max(axis=0))
                    img = downscale_if_needed(img)
                    imgs.append(img)
                    names.append("cells3d_proj")
                except Exception:
                    img = to_float_gray(data.camera())
                    img = downscale_if_needed(img)
                    imgs.append(img)
                    names.append("camera")

    return imgs, names


def initial_ellipse_mask(shape, frac=0.35):
    """Sukuria elipsės bool maską, kurios ašys ~ frac * min(H, W)."""
    H, W = shape
    r0 = int(H / 2)
    c0 = int(W / 2)
    a = int(frac * W)
    b = int(frac * H)
    rr, cc = draw_ellipse(r0, c0, b, a, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def initial_ring_mask(shape, inner_frac=0.20, outer_frac=0.45):
    H, W = shape
    r0 = int(H / 2)
    c0 = int(W / 2)
    rr_o, cc_o = draw_disk((r0, c0), int(outer_frac * min(H, W)), shape=shape)
    rr_i, cc_i = draw_disk((r0, c0), int(inner_frac * min(H, W)), shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr_o, cc_o] = True
    mask[rr_i, cc_i] = False
    return mask


def initial_snake_ellipse(shape, frac=0.45, n_points=200):
    H, W = shape
    r0 = H / 2.0
    c0 = W / 2.0
    a = frac * W
    b = frac * H
    s = np.linspace(0, 2 * np.pi, n_points)
    # active_contour tikisi (row, col): [y, x]
    init = np.column_stack([r0 + b * np.sin(s), c0 + a * np.cos(s)])
    return init


def iou_dice(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = inter / union if union > 0 else 0.0
    dice = 2 * inter / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0.0
    return iou, dice


def area_mean_intensity(img, mask):
    mask = mask.astype(bool)
    area = int(mask.sum())
    mean_int = float(img[mask].mean()) if area > 0 else 0.0
    return area, mean_int

# Pašalina smulkias saleles ir pasilieka didžiausią komponentą – kad CV/MGAC neperdėtų ploto
def clean_mask(mask, img_shape, min_frac=0.003):
    m = mask.astype(bool)
    h, w = img_shape
    min_size = int(min_frac * h * w)
    m = remove_small_objects(m, min_size=min_size)
    m = binary_closing(m, footprint=disk(2))
    lab, n = ndimage.label(m)
    if n > 0:
        sizes = ndimage.sum(m, lab, index=range(1, n+1))
        keep_label = int(np.argmax(sizes)) + 1
        m = (lab == keep_label)
    return m


# ---------- Metodai ----------

def run_snakes(img, alpha=0.01, beta=1.0, gamma=0.1, smooth_sigma=1.0, init_frac=0.45):
    img_s = gaussian(img, sigma=smooth_sigma)
    init = initial_snake_ellipse(img.shape, frac=init_frac)
    snake = active_contour(
        img_s,
        init,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        max_num_iter=N_ITER_SNAKE,
    )

    rr, cc = np.round(snake[:, 0]).astype(int), np.round(snake[:, 1]).astype(int)
    rr = np.clip(rr, 0, img.shape[0] - 1)
    cc = np.clip(cc, 0, img.shape[1] - 1)
    mask = np.zeros_like(img, dtype=bool)
    mask[rr, cc] = True

    # Užpildome visą gyvatės poligoną, kad plotas būtų tikras objekto plotas
    rrf, ccf = draw_polygon(rr, cc, img.shape)
    mask_poly = np.zeros_like(img, dtype=bool)
    mask_poly[rrf.astype(int), ccf.astype(int)] = True
    return mask_poly, snake

def run_chan_vese(img, num_iter=N_ITER_CV, smoothing=3, init_frac=0.45):
    mask0 = initial_ellipse_mask(img.shape, frac=init_frac)
    res = morphological_chan_vese(img, num_iter=num_iter, init_level_set=mask0, smoothing=smoothing)
    return res.astype(bool), mask0


def run_mgac(img, num_iter=N_ITER_MGAC, smoothing=1, alpha=100.0, sigma=1.0, balloon=0.5, inner_frac=0.20, outer_frac=0.45):
    gimg = inverse_gaussian_gradient(img, alpha=alpha, sigma=sigma)
    init = initial_ring_mask(img.shape, inner_frac=inner_frac, outer_frac=outer_frac)
    res = morphological_geodesic_active_contour(
        gimg,
        num_iter=num_iter,
        init_level_set=init,
        smoothing=smoothing,
        threshold='auto',
        balloon=balloon,
    )
    return res.astype(bool), init


# ---------- Bėgimas vienam vaizdui + vizualizacijos ----------

def run_all_on_image(img, name):
    results = {}

    # 1) Originalas
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.title(f"Originalas – {name}")
    plt.axis('off')
    save_fig(fig, f"{PFX_S1}{name}_original.png")

    # 2) Snakes
    snake_mask, snake_points = run_snakes(img)
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.contour(snake_mask, colors='r', linewidths=1)
    plt.title("Snakes (kraštinis)")
    plt.axis('off')
    save_fig(fig, f"{PFX_S2}{name}_snakes.png")
    results['snakes'] = snake_mask

    # 3) Chan–Vese (regioninis)
    cv_mask, cv_init = run_chan_vese(img)
    cv_mask = clean_mask(cv_mask, img.shape)
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.contour(cv_mask, colors='y', linewidths=1)
    plt.title("Chan–Vese (regioninis)")
    plt.axis('off')
    save_fig(fig, f"{PFX_S3}{name}_chanvese.png")
    results['chanvese'] = cv_mask

    # 4) MGAC (geodezinis)
    params = pick_mgac_params(name)
    mgac_mask, mgac_init = run_mgac(img, **params)
    mgac_mask = clean_mask(mgac_mask, img.shape)
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.contour(mgac_mask, colors='c', linewidths=1)
    plt.title("MGAC (geodezinis)")
    plt.axis('off')
    save_fig(fig, f"{PFX_S4}{name}_mgac.png")
    results['mgac'] = mgac_mask

    # 5) Proxy metrikos
    metrics = []
    for method, mask in results.items():
        area, mean_int = area_mean_intensity(img, mask)
        metrics.append({"image": name, "method": method, "area_px": area, "mean_intensity": mean_int})
    # Kombinuotas vaizdas su visais metodais
    save_overlay_multi(
        img,
        results,
        title=f"[5] Combined – {name}",
        path=f"{PFX_S5}{name}_combined.png",
        color_map={"snakes": "r", "chanvese": "y", "mgac": "c"}
    )
    return results, pd.DataFrame(metrics)


# ---------- Triukšmas ir inicializacija ----------

def noise_and_init_study(img, name):
    from skimage.util import random_noise

    noisy = random_noise(img, mode='gaussian', var=0.01)
    den_med = median(noisy, footprint=disk(3))
    den_gauss = gaussian(noisy, sigma=1)

    # 1) Parodyti triukšmo įtaką snakes/chanvese/mgac (naudojame tas pačias funkcijas)
    for variant, arr in [("noisy", noisy), ("den_med", den_med), ("den_gauss", den_gauss)]:
        masks, _ = run_all_on_image(arr, f"{name}_{variant}")
        # virš originalo – palyginimo paveikslas
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        for col, (mname, msk) in zip(['r', 'y', 'c'], masks.items()):
            plt.contour(msk, colors=col, linewidths=1)
        plt.title(f"Palyginimas ant originalo – {variant}")
        plt.axis('off')
        save_fig(fig, f"{name}_{variant}_compare_on_clean.png")

    # 2) Inicializacijos pokyčiai: snakes su mažesne/didesne pradine elipse
    for frac, tag in [(0.30, "smaller"), (0.60, "larger")]:
        mask, _ = run_snakes(img, init_frac=frac)
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap='gray')
        plt.contour(mask, colors='m', linewidths=1)
        plt.title(f"Snakes – init {tag}")
        plt.axis('off')
        save_fig(fig, f"{name}_snakes_init_{tag}.png")


def aggregate_bar(df_all):
    # Bar grafikas – vidutinis plotas pagal metodą
    df_mean = df_all.groupby('method', as_index=False)['area_px'].mean()
    fig = plt.figure(figsize=(6, 4))
    plt.bar(df_mean['method'], df_mean['area_px'])
    plt.title('Vidutinis užimtas plotas (px) pagal metodą')
    plt.xlabel('Metodas')
    plt.ylabel('Plotas (px)')
    save_fig(fig, '99_bar_plot_area.png')


# ---------- Papildoma A) Seed region + Chan–Vese ----------
from skimage.draw import disk as draw_disk

def run_seed_chanvese(img, seed_radius_frac=0.08):
    h, w = img.shape
    r0, c0 = h//2, w//2
    rr, cc = draw_disk((r0, c0), int(seed_radius_frac*min(h,w)), shape=img.shape)
    seed = np.zeros(img.shape, dtype=bool); seed[rr, cc] = True
    m = morphological_chan_vese(img, num_iter=N_ITER_CV, init_level_set=seed, smoothing=2)
    return m.astype(bool), seed


# ---------- Papildoma B) Keli objektai su MGAC ----------

def run_multi_mgac(img, grid_step=80, seed_radius=10, alpha=100.0, sigma=1.0, balloon=0.8):
    h, w = img.shape
    seeds = np.zeros((h, w), dtype=bool)
    for r in range(seed_radius, h, grid_step):
        for c in range(seed_radius, w, grid_step):
            rr, cc = draw_disk((r, c), seed_radius, shape=img.shape)
            seeds[rr, cc] = True
    gimg = inverse_gaussian_gradient(img, alpha=alpha, sigma=sigma)
    m = morphological_geodesic_active_contour(
        gimg, num_iter=N_ITER_MGAC, init_level_set=seeds, smoothing=1, threshold='auto', balloon=balloon
    )
    return m.astype(bool), seeds


def main():
    imgs, names = load_demo_images()

    # 1–5 žingsniai: paleidžiame tris metodus ant bent dviejų vaizdų
    dfs = []
    for img, name in zip(imgs[:2], names[:2]):  # du vaizdai kaip minimumas
        res, dfm = run_all_on_image(img, name)
        dfs.append(dfm)
        if RUN_NOISE:
            noise_and_init_study(img, name)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(OUT_DIR / 'metrics.csv', index=False)
    aggregate_bar(df_all)

    # Papildoma A) – Seed region + Chan–Vese
    for img, name in zip(imgs[:2], names[:2]):
        mA, seedA = run_seed_chanvese(img)
        fig = plt.figure(figsize=(5,5))
        plt.imshow(img, cmap='gray'); plt.contour(mA, colors='g', linewidths=1)
        plt.title(f"[A] Seed Chan–Vese – {name}")
        plt.axis('off')
        save_fig(fig, f"{PFX_A}{name}_seed_chanvese.png")

    # Papildoma B) – Keli objektai su MGAC
    for img, name in zip(imgs[:2], names[:2]):
        p = pick_mgac_params(name)
        mB, seeds = run_multi_mgac(img, alpha=p["alpha"], sigma=p["sigma"], balloon=p["balloon"])
        fig = plt.figure(figsize=(5,5))
        plt.imshow(img, cmap='gray'); plt.contour(mB, colors='m', linewidths=1)
        plt.title(f"[B] Multi-object MGAC – {name}")
        plt.axis('off')
        save_fig(fig, f"{PFX_B}{name}_multi_mgac.png")

    # 5–8 sakinių išvados – minimalus šablonas (išsisaugome į txt)
    conclusions = (
        "1) Kraštinis (Snakes) jautresnis glotninimui ir pradinės kreivės padėčiai: per toli esant – gali nepritraukti.\n"
        "2) Regioninis (Chan–Vese) geriau veikia su neryškiais kraštais ir netolygiu apšvietimu, bet gali 'nutekėti' į foną jei klasės panašios.\n"
        "3) MGAC su 'balloon'>0 plečia ribas ir stabilesnis prie triukšmo, tačiau reikia tinkamai parinkti sigma/alpha potencialui.\n"
        "4) Triukšmas labiausiai kenkia kraštiniam metodui; median/gaussian filtrai pagerina konvergenciją.\n"
        "5) Didesnis glotnumas (beta/smoothing) padeda vengti dantytų ribų, bet gali perkelti kontūrą nuo tikslios sienos.\n"
        "6) Regioninį rinktis, kai kraštai silpni ir objektas skiriasi intensyvumu; kraštinį – kai ribos aiškios. MGAC – kai reikia balansuoti tarp abiejų.\n"
    )
    (OUT_DIR / 'conclusions.txt').write_text(conclusions, encoding='utf-8')

    print(f"\nRezultatai: {OUT_DIR.resolve()}")
    print("Sekcijos: 1_* (originalai), 2_* (Snakes), 3_* (Chan–Vese), 4_* (MGAC), 5_* (metrikos)")
    print("Papildoma: A_* (Seed+Chan–Vese), B_* (Multi-object MGAC)")


if __name__ == "__main__":
    main()