from pathlib import Path

# --- Output folder (saves all Lab7 results here) ---
OUTPUT_DIR = Path(__file__).resolve().parent / "lab7_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Jupyter display fallback (for PyCharm / terminal runs) ---
try:
    from IPython.display import display  # type: ignore
except Exception:
    def display(x):
        print(x)

# Global counter for figure filenames
_FIG_COUNTER = 0

def save_fig(fig, stem: str):
    """Save matplotlib figure into OUTPUT_DIR with an auto-incremented filename."""
    global _FIG_COUNTER
    _FIG_COUNTER += 1
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    out = OUTPUT_DIR / f"{_FIG_COUNTER:02d}_{safe_stem}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[SAVED] {out}")
    return out

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.exposure import equalize_hist
from skimage.filters import sobel, threshold_otsu, gaussian
from skimage.segmentation import watershed, find_boundaries
from skimage.morphology import closing, square, remove_small_objects, remove_small_holes
from skimage.feature import peak_local_max
from skimage.util import random_noise
from scipy import ndimage as ndi

# -----------------------------
# 0) Pagalbinės funkcijos
# -----------------------------
def prep_image(img, do_equalize=True):
    """Normalizuoja į float [0,1], opcionaliai equalize_hist kontrastui."""
    img_f = img_as_float(img)
    if do_equalize:
        img_f = equalize_hist(img_f)
    return img_f

def make_binary_and_distance(img_f, invert="auto",
                             close_size=3,
                             remove_small=64,
                             fill_small_holes=64):
    """
    Binarizacija (Otsu) + morfologija + distance transform.
    invert:
      - "auto": pasirenka ar objektai tamsūs ar šviesūs (pagal medianą)
      - True/False: priverstinai
    """
    th = threshold_otsu(img_f)

    if invert == "auto":
        # jei dauguma pikselių šviesūs, tai tamsūs objektai -> img_f < th
        # jei dauguma tamsūs, tai šviesūs objektai -> img_f > th
        invert = (np.median(img_f) > th)

    bw = (img_f < th) if invert else (img_f > th)

    bw_clean = closing(bw, square(close_size))
    bw_clean = remove_small_objects(bw_clean, min_size=remove_small)
    bw_clean = remove_small_holes(bw_clean, area_threshold=fill_small_holes)

    dist = ndi.distance_transform_edt(bw_clean)
    return bw_clean, dist, th, invert

def make_markers_from_distance(dist, bw_clean, footprint=25, min_distance=1):
    """
    Foreground seed'ai per peak_local_max -> markers per connected components.
    footprint stipriai lemia seed'ų (objektų) skaičių.
    """
    fp = np.ones((footprint, footprint), dtype=bool)
    coords = peak_local_max(dist, footprint=fp, labels=bw_clean, min_distance=min_distance)
    local_maxi = np.zeros_like(dist, dtype=bool)
    if coords.size > 0:
        local_maxi[coords[:, 0], coords[:, 1]] = True
    markers = ndi.label(local_maxi)[0]
    return markers, coords

def segment_all_methods(img_f, bw_clean, dist, markers):
    """3 metodai: klasikinė, marker-based (dist), gradient-based (sobel)."""
    grad = sobel(img_f)

    # 1) Klasikinė watershed tiesiog gradientui (be markers) -> persegmentavimas
    labels_classic = watershed(grad)

    # 2) Marker-based watershed ant -dist (gerai atskiria apvalius objektus)
    labels_marker = watershed(-dist, markers, mask=bw_clean)

    # 3) Gradient-based marker-controlled watershed (geresnės ribos, bet jautresnė triukšmui)
    labels_grad = watershed(grad, markers, mask=bw_clean)

    return labels_classic, labels_marker, labels_grad, grad

def label_to_object_mask(labels, bw_clean=None):
    """
    Konvertuoja label'ius į objektų maską.
    - Classic atveju be mask: imame ne-0 regionus (bet 0 gali nebūti); saugiau:
      naudojame boundary pagrindu? paprasčiau: (labels > 0).
    - Marker/Grad atveju: objektai ten, kur mask bw_clean.
    """
    if bw_clean is None:
        return labels > 0
    else:
        return (labels > 0) & bw_clean

def metrics_no_gt(img_f, obj_mask, labels=None):
    """Jei nėra GT: plotas, mean intensity objekte, objektų skaičius."""
    area = int(obj_mask.sum())
    mean_int = float(img_f[obj_mask].mean()) if area > 0 else 0.0

    if labels is None:
        # komponentai iš mask
        n_obj = int(ndi.label(obj_mask)[1])
    else:
        # objektų skaičius = unikalių label (be 0)
        u = np.unique(labels)
        n_obj = int((u != 0).sum())

    return area, mean_int, n_obj

def overlay_boundaries(ax, img_f, labels, color='red', title=''):
    ax.imshow(img_f, cmap='gray')
    b = find_boundaries(labels, mode='outer')
    ax.contour(b, colors=color, linewidths=0.8)
    ax.set_title(title)
    ax.axis('off')

# -----------------------------
# 1) Įkelk 2 vaizdus + normalizavimas
# -----------------------------
img1 = data.coins()  # monetos

# Antras vaizdas: bandome cells3d (reikalauja optional priklausomybės `pooch`).
# Jei jos nėra – naudojame kitą pilkumo vaizdą iš skimage (moon).
try:
    cells = data.cells3d()  # (z, y, x, channels)
    img2 = cells[30, :, :, 1]  # vienas pjūvis, kanalą gali pakeisti 0/1
except ModuleNotFoundError as e:
    print(
        "[INFO] data.cells3d() nepavyko (trūksta optional dependency 'pooch'). "
        "Naudoju data.moon() kaip antrą vaizdą.\n"
        "Jei nori cells3d: pip install pooch"
    )
    img2 = data.moon()

img1_f = prep_image(img1, do_equalize=True)
img2_f = prep_image(img2, do_equalize=True)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(img1_f, cmap='gray'); ax[0].set_title('coins (normalized + equalize)'); ax[0].axis('off')
ax[1].imshow(img2_f, cmap='gray'); ax[1].set_title('cells3d slice (normalized + equalize)'); ax[1].axis('off')
plt.tight_layout()
save_fig(fig, "01_inputs_normalized")
plt.show()

# -----------------------------
# 2–4) Segmentacija abiem vaizdams
# -----------------------------
def run_pipeline_for_image(img_f, name,
                           invert="auto",
                           close_size=3,
                           remove_small=64,
                           fill_small_holes=64,
                           footprint=25):
    bw_clean, dist, th, inv_used = make_binary_and_distance(
        img_f, invert=invert,
        close_size=close_size, remove_small=remove_small, fill_small_holes=fill_small_holes
    )
    markers, coords = make_markers_from_distance(dist, bw_clean, footprint=footprint)
    labels_classic, labels_marker, labels_grad, grad = segment_all_methods(img_f, bw_clean, dist, markers)

    # Vizualizacijos: 1–2 per metodą (čia: po 1 kiekvienam metodui + papildomai classic segmentų žemėlapis)
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    ax[0,0].imshow(img_f, cmap='gray'); ax[0,0].set_title(f'{name}: original'); ax[0,0].axis('off')
    ax[0,1].imshow(grad, cmap='gray'); ax[0,1].set_title('Sobel gradient'); ax[0,1].axis('off')
    ax[0,2].imshow(bw_clean, cmap='gray'); ax[0,2].set_title(f'BW clean (Otsu, invert={inv_used})'); ax[0,2].axis('off')

    ax[1,0].imshow(labels_classic, cmap='nipy_spectral'); ax[1,0].set_title('Classic watershed (no markers)'); ax[1,0].axis('off')
    overlay_boundaries(ax[1,1], img_f, labels_marker, color='red', title=f'Marker-based watershed (fp={footprint})')
    overlay_boundaries(ax[1,2], img_f, labels_grad, color='yellow', title='Gradient-based watershed')

    plt.tight_layout()
    save_fig(fig, f"{name}_02_methods")
    plt.show()

    # Metrikos be GT (plotas, mean intensity, objektų skaičius)
    obj_classic = label_to_object_mask(labels_classic)  # classic be mask
    obj_marker  = label_to_object_mask(labels_marker, bw_clean=bw_clean)
    obj_grad    = label_to_object_mask(labels_grad, bw_clean=bw_clean)

    a1, m1, n1 = metrics_no_gt(img_f, obj_classic, labels_classic)
    a2, m2, n2 = metrics_no_gt(img_f, obj_marker,  labels_marker)
    a3, m3, n3 = metrics_no_gt(img_f, obj_grad,    labels_grad)

    out = {
        "name": name,
        "th_otsu": float(th),
        "invert_used": bool(inv_used),
        "markers_count": int(markers.max()),
        "classic_area": a1, "classic_mean_int": m1, "classic_nobj": n1,
        "marker_area": a2,  "marker_mean_int": m2,  "marker_nobj": n2,
        "grad_area": a3,    "grad_mean_int": m3,    "grad_nobj": n3,
        "bw_clean": bw_clean, "dist": dist, "markers": markers,
        "labels_classic": labels_classic, "labels_marker": labels_marker, "labels_grad": labels_grad
    }
    return out

res1 = run_pipeline_for_image(img1_f, "coins", invert="auto", footprint=25)
res2 = run_pipeline_for_image(img2_f, "cells3d", invert="auto", footprint=21)

# -----------------------------
# 5) Palyginimo lentelė + bar chart
# -----------------------------
rows = []
for r in [res1, res2]:
    rows.append([r["name"], "classic", r["classic_area"], r["classic_mean_int"], r["classic_nobj"]])
    rows.append([r["name"], "marker",  r["marker_area"],  r["marker_mean_int"],  r["marker_nobj"]])
    rows.append([r["name"], "grad",    r["grad_area"],    r["grad_mean_int"],    r["grad_nobj"]])

df = pd.DataFrame(rows, columns=["image", "method", "area(px)", "mean_intensity", "n_objects"])
display(df)
df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
print(f"[SAVED] {OUTPUT_DIR / 'metrics_summary.csv'}")

# Bar chart: objektų skaičius (galima pakeist į area arba mean_intensity)
fig, ax = plt.subplots(figsize=(8, 4))
pivot = df.pivot(index="method", columns="image", values="n_objects")
pivot.plot(kind="bar", ax=ax)
ax.set_ylabel("Object count")
ax.set_title("Watershed methods comparison (no GT): number of objects")
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
save_fig(fig, "03_bar_n_objects")
plt.show()

# -----------------------------
# 6A) Triukšmas + filtrai (median/gaussian) ir jautrumas
# -----------------------------
from skimage.filters import median
from skimage.morphology import disk

def noise_filter_experiment(img_f, name, footprint=25, noise_var=0.01):
    noisy = random_noise(img_f, mode='gaussian', var=noise_var)
    gaus = gaussian(noisy, sigma=1.0)
    med  = median(noisy, footprint=disk(2))

    variants = [
        ("clean", img_f),
        ("noisy", noisy),
        ("gaussian_filtered", gaus),
        ("median_filtered", med),
    ]

    results = []
    for vname, im in variants:
        bw_clean, dist, th, inv_used = make_binary_and_distance(im, invert="auto")
        markers, _ = make_markers_from_distance(dist, bw_clean, footprint=footprint)
        labels_classic, labels_marker, labels_grad, _ = segment_all_methods(im, bw_clean, dist, markers)

        # Metrika: objektų skaičius (stabilumas) + plotas
        obj_classic = label_to_object_mask(labels_classic)
        obj_marker  = label_to_object_mask(labels_marker, bw_clean=bw_clean)
        obj_grad    = label_to_object_mask(labels_grad, bw_clean=bw_clean)

        a1, _, n1 = metrics_no_gt(im, obj_classic, labels_classic)
        a2, _, n2 = metrics_no_gt(im, obj_marker,  labels_marker)
        a3, _, n3 = metrics_no_gt(im, obj_grad,    labels_grad)

        results.append([name, vname, "classic", a1, n1])
        results.append([name, vname, "marker",  a2, n2])
        results.append([name, vname, "grad",    a3, n3])

    df_noise = pd.DataFrame(results, columns=["image", "variant", "method", "area(px)", "n_objects"])
    display(df_noise)
    out_csv = OUTPUT_DIR / f"noise_metrics_{name}.csv"
    df_noise.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")

    # Vizualiai: noisy vs filtruoti + grad-based ribos
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for i, (vname, im) in enumerate(variants):
        bw_clean, dist, _, _ = make_binary_and_distance(im, invert="auto")
        markers, _ = make_markers_from_distance(dist, bw_clean, footprint=footprint)
        labels_classic, labels_marker, labels_grad, _ = segment_all_methods(im, bw_clean, dist, markers)
        overlay_boundaries(ax[i], im, labels_grad, color='yellow', title=f'{vname}: grad-ws')
    plt.tight_layout()
    save_fig(fig, f"{name}_04_noise_grad_ws")
    plt.show()

    return df_noise

df_noise1 = noise_filter_experiment(img1_f, "coins", footprint=25, noise_var=0.01)

# -----------------------------
# 6B) Marker’ių įtaka: per mažai / per daug (footprint keitimas)
# -----------------------------
def marker_sensitivity_demo(img_f, name, footprints=(9, 21, 41)):
    bw_clean, dist, th, inv_used = make_binary_and_distance(img_f, invert="auto")

    fig, ax = plt.subplots(1, len(footprints), figsize=(5*len(footprints), 4))
    if len(footprints) == 1:
        ax = [ax]

    for i, fp in enumerate(footprints):
        markers, coords = make_markers_from_distance(dist, bw_clean, footprint=fp)
        labels = watershed(-dist, markers, mask=bw_clean)
        overlay_boundaries(ax[i], img_f, labels, color='red', title=f'{name}: fp={fp}, markers={markers.max()}')
    plt.tight_layout()
    save_fig(fig, f"{name}_05_marker_sensitivity")
    plt.show()

marker_sensitivity_demo(img1_f, "coins", footprints=(9, 21, 41))

# -----------------------------
# Išvados (automatiškai sugeneruota santrauka pagal tipinius stebėjimus)
# -----------------------------
print("IŠVADOS (įrašyk į ataskaitą 5–8 sakinius, gali adaptuoti):")
print("1) Klasikinė watershed (be žymeklių) dažniausiai persegmentuoja, nes kiekvienas mažas gradiento minimumas tampa atskiru baseinu.")
print("2) Marker-based watershed stabilizuoja segmentaciją: objektų skaičius labiau atitinka realius objektus, ypač kai objektai atskiriami per distance transform.")
print("3) Gradient-based marker-controlled watershed dažnai duoda tikslesnes ribas, bet triukšmas padidina gradientų netolygumą, todėl metodas tampa jautresnis be filtravimo.")
print("4) Keičiant peak_local_max footprint: per mažas footprint -> per daug seed'ų -> objektai 'susprogsta' į daug segmentų; per didelis footprint -> per mažai seed'ų -> objektai susijungia.")
print("5) Triukšmui dažniausiai jautriausia gradient-based schema; median/gaussian filtrai sumažina klaidingų ribų skaičių ir stabilizuoja seed'ų paiešką.")
print("6) Praktikoje: jei prioritetas – teisingas objektų skaičius, rinktis marker-based; jei prioritetas – tikslios ribos ir yra filtravimas, rinktis gradient-based.")

conclusions_lines = [
    "IŠVADOS:",
    "1) Klasikinė watershed (be žymeklių) dažniausiai persegmentuoja, nes kiekvienas mažas gradiento minimumas tampa atskiru baseinu.",
    "2) Marker-based watershed stabilizuoja segmentaciją: objektų skaičius labiau atitinka realius objektus, ypač kai objektai atskiriami per distance transform.",
    "3) Gradient-based marker-controlled watershed dažnai duoda tikslesnes ribas, bet triukšmas padidina gradientų netolygumą, todėl metodas tampa jautresnis be filtravimo.",
    "4) Keičiant peak_local_max footprint: per mažas footprint -> per daug seed'ų -> objektai 'susprogsta' į daug segmentų; per didelis footprint -> per mažai seed'ų -> objektai susijungia.",
    "5) Triukšmui dažniausiai jautriausia gradient-based schema; median/gaussian filtrai sumažina klaidingų ribų skaičių ir stabilizuoja seed'ų paiešką.",
    "6) Praktikoje: jei prioritetas – teisingas objektų skaičius, rinktis marker-based; jei prioritetas – tikslios ribos ir yra filtravimas, rinktis gradient-based.",
]
(OUTPUT_DIR / "conclusions.txt").write_text("\n".join(conclusions_lines), encoding="utf-8")
print(f"[SAVED] {OUTPUT_DIR / 'conclusions.txt'}")

# Small README for your submission
readme = (
    "Lab7 output saved automatically.\n\n"
    "PNG figures:\n"
    "- 01_inputs_normalized.png\n"
    "- <image>_02_methods.png (classic / marker / gradient overlays)\n"
    "- 03_bar_n_objects.png\n"
    "- <image>_04_noise_grad_ws.png\n"
    "- <image>_05_marker_sensitivity.png\n\n"
    "Tables:\n"
    "- metrics_summary.csv\n"
    "- noise_metrics_<image>.csv\n\n"
    "Text:\n"
    "- conclusions.txt\n"
)
(OUTPUT_DIR / "README.txt").write_text(readme, encoding="utf-8")
print(f"[SAVED] {OUTPUT_DIR / 'README.txt'}")