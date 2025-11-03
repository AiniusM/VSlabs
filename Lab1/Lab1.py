from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

# === 1. Nustatymai ===
IMAGE_PATH = r"/Users/ainius/PycharmProjects/VSlabs/Lab1/data/nuot.jpeg"
x, y = 50, 50                 # pikselio taškas (x, y) – gali keisti
roi_x1, roi_y1 = 30, 30       # ROI viršutinis-kairys kampas
roi_x2, roi_y2 = 130, 130     # ROI dešinys-apatinis kampas (neįskaitytinai)

# Kur bus saugomi paveikslai
OUT_DIR = Path("lab1_outputs")
OUT_DIR.mkdir(exist_ok=True)

# Pagalbinė funkcija: išsaugo ir uždaro figūrą (be rodymo)
def save_fig(fig, name):
    out_path = OUT_DIR / name
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Išsaugota: {out_path}")

# === 2. Įkėlimas ===
p = Path(IMAGE_PATH)
if not p.exists():
    raise FileNotFoundError(f"Nerastas vaizdo failas: {p.resolve()}")

img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
if img_bgr is None:
    raise RuntimeError("Nepavyko perskaityti vaizdo. Patikrink, ar failas nėra sugadintas.")

# Bazine info
h, w = img_bgr.shape[:2]
channels = img_bgr.shape[2] if img_bgr.ndim == 3 else 1
bit_depth = img_bgr.itemsize * 8
print("=== 1) Įkėlimas ir bazinė informacija ===")
print(f"Failas: {p.name}")
print(f"Dydis (Plotis x Aukštis): {w} x {h}")
print(f"Kanalų sk.: {channels}")
print(f"Bitų gylis/kanalui: {bit_depth}")

# Teisingas atvaizdavimas (matplotlib tikisi RGB, o cv2 pateikia BGR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
fig1 = plt.figure()
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Originalas (RGB)")
save_fig(fig1, "01_originalas_rgb.png")

# === 3. Spalvų kanalai (R, G, B) – VIENAME PAVEIKSLE ===
B, G, R = cv2.split(img_bgr)   # BGR tvarka

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(R, cmap="gray"); axes[0].set_title("R kanalas"); axes[0].axis("off")
axes[1].imshow(G, cmap="gray"); axes[1].set_title("G kanalas"); axes[1].axis("off")
axes[2].imshow(B, cmap="gray"); axes[2].set_title("B kanalas"); axes[2].axis("off")
fig.suptitle("R/G/B kanalai (pilkumo režimu)")
plt.tight_layout()
save_fig(fig, "02_kanalai_rgb.png")

print("Pastaba: šviesesnės vietos kanale reiškia didesnę to kanalo įtaką tose vaizdo vietose.")

# === 4. Konvertavimas į grayscale ===
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
fig3 = plt.figure()
plt.imshow(gray, cmap="gray")
plt.axis("off")
plt.title("Grayscale")
save_fig(fig3, "03_grayscale.png")
print("Informacija: iš 3 kanalų (R/G/B) → 1 kanalas (grayscale). Prarandam spalvą, lieka šviesumas.")

# === 5. Pikselių prieiga ir ROI ===
# suklampinam, kad neišeitų už ribų
x = max(0, min(w - 1, x))
y = max(0, min(h - 1, y))
print("\n=== 4) Pikselių prieiga ir ROI statistika ===")
print(f"Pikselio BGR ({x}, {y}): {img_bgr[y, x].tolist()}")
print(f"Pikselio GRAY ({x}, {y}): {int(gray[y, x])}")

x1 = max(0, min(w - 1, roi_x1)); y1 = max(0, min(h - 1, roi_y1))
x2 = max(x1 + 1, min(w, roi_x2)); y2 = max(y1 + 1, min(h, roi_y2))
roi_color = img_bgr[y1:y2, x1:x2]
roi_gray  = gray[y1:y2, x1:x2]

roi_gray_mean = float(np.mean(roi_gray))
roi_gray_std  = float(np.std(roi_gray))
roi_B_mean, roi_G_mean, roi_R_mean = [float(np.mean(roi_color[:, :, i])) for i in range(3)]
roi_B_std,  roi_G_std,  roi_R_std  = [float(np.std(roi_color[:, :, i])) for i in range(3)]

print(f"ROI [{x1}:{x2}, {y1}:{y2}] dydis: {roi_color.shape[1]}x{roi_color.shape[0]} px")
print(f"Grayscale ROI mean/std: {roi_gray_mean:.2f} / {roi_gray_std:.2f}")
print(f"B kanalo mean/std: {roi_B_mean:.2f} / {roi_B_std:.2f}")
print(f"G kanalo mean/std: {roi_G_mean:.2f} / {roi_G_std:.2f}")
print(f"R kanalo mean/std: {roi_R_mean:.2f} / {roi_R_std:.2f}")

fig4 = plt.figure()
plt.imshow(roi_gray, cmap="gray")
plt.axis("off")
plt.title("ROI (grayscale)")
save_fig(fig4, "04_roi_grayscale.png")

# === 6. Histogramos – VISOS VIENAME GRAFIKE ===
plt.figure(figsize=(9, 5))
plt.hist(gray.ravel(), bins=256, range=(0, 256), alpha=0.7, label="Grayscale", color="black")
plt.hist(R.ravel(),    bins=256, range=(0, 256), alpha=0.4, label="R", color="red")
plt.hist(G.ravel(),    bins=256, range=(0, 256), alpha=0.4, label="G", color="green")
plt.hist(B.ravel(),    bins=256, range=(0, 256), alpha=0.4, label="B", color="blue")
plt.title("Histogramos: Grayscale + R/G/B viename grafike")
plt.xlabel("Intensyvumas (0–255)")
plt.ylabel("Dažnis")
plt.legend()
plt.tight_layout()
save_fig(plt.gcf(), "05_histogramos.png")

# Paprasta užuomina apie formą (vienkalnė/dviekalnė)
hist_counts, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
peaks = sum(1 for i in range(1, 255) if hist_counts[i] > hist_counts[i-1] and hist_counts[i] > hist_counts[i+1])
form_hint = "gali būti dviekalnė/mišri" if peaks > 4 else "link vienkalnės"
print(f"Histogramos forma (grubiai): {form_hint}")

# === 7. Formatai ir failo dydžiai ===
out_png = OUT_DIR / "isvestis_png.png"
out_j95 = OUT_DIR / "isvestis_q95.jpg"
out_j60 = OUT_DIR / "isvestis_q60.jpg"

cv2.imwrite(str(out_png), img_bgr)                          # PNG (lossless)
cv2.imwrite(str(out_j95), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])  # JPEG Q=95
cv2.imwrite(str(out_j60), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])  # JPEG Q=60

size_png = out_png.stat().st_size / 1024
size_j95 = out_j95.stat().st_size / 1024
size_j60 = out_j60.stat().st_size / 1024
print("\n=== 6) Failų dydžiai ===")
print(f"PNG: {size_png:.1f} KB | JPEG Q95: {size_j95:.1f} KB | JPEG Q60: {size_j60:.1f} KB")
print("Komentaras: JPEG – nuostolingas (mažesni failai, artefaktai su žemesne kokybe); PNG – nenuostolingas (tikslūs pikseliai).")
print(f"Visi paveikslai ir eksportai išsaugoti aplanke: {OUT_DIR.resolve()}")

# === 8. Santrauka ataskaitai (greiti atsakymai) ===
mem_kb = w * h * channels * (bit_depth // 8) / 1024
print("\n=== Santrauka (į ataskaitą) ===")
print("• RGB/BGR vs. Grayscale: RGB/BGR turi 3 kanalus (spalva), grayscale – 1 (šviesumas). Pereinant į grayscale prarandama spalva.")
print(f"• Rezoliucija ir bitų gylis: {w}×{h}, {bit_depth} bit/kanalui, {channels} kanalai → ~{mem_kb:.1f} KB nekompresuotos atminties.")
print(f"• Histogramos forma: {form_hint}. Plati/lygiai pasiskirsčiusi histograma → didesnis kontrastas; siaura → mažesnis.")
print("• PNG vs. JPEG: PNG – nenuostolingas (gerai grafikams/tekstui), JPEG – nuostolingas (gerai fotografijoms, mažesni failai).")
