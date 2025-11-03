import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Kur bus saugomi paveikslai (Lab2)
OUT_DIR = Path("lab2_outputs")
OUT_DIR.mkdir(exist_ok=True)

# Pagalbinė funkcija: išsaugo ir uždaro figūrą (be rodymo)
def save_fig(fig, name):
    out_path = OUT_DIR / name
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Išsaugota: {out_path}")

# 1. Įkėlimas pilkumo režimu
gray = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Patikrinimas
fig1 = plt.figure(figsize=(6, 4))
plt.imshow(gray, cmap='gray')
plt.title('Originalus pilkumo vaizdas')
plt.axis('off')
save_fig(fig1, "01_originalus_pilkumo.png")

# Histogramos braižymas
fig2 = plt.figure(figsize=(6, 4))
plt.hist(gray.ravel(), bins=256, range=(0, 256))
plt.title('Pilkumo histograma')
plt.xlabel('Intensyvumas')
plt.ylabel('Pikselių skaičius')
save_fig(fig2, "02_histograma.png")

# 2. Globalus (fiksuotas) slenkstis
# Trys skirtingi slenksčiai
for T in [100, 128, 180]:
    _, binary = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
    fig_fixed = plt.figure()
    plt.imshow(binary, cmap='gray')
    plt.title(f'Fiksuotas slenkstis T = {T}')
    plt.axis('off')
    save_fig(fig_fixed, f"03_fiksuotas_T{T}.png")

# 3. Automatinis (Otsu) metodas
T_otsu, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Otsu parinktas slenkstis:", T_otsu)

fig3 = plt.figure()
plt.imshow(binary_otsu, cmap='gray')
plt.title(f'Otsu metodas (T = {T_otsu:.0f})')
plt.axis('off')
save_fig(fig3, "06_otsu.png")

# 4. Lokalus (adaptuojami) metodai
# Adaptuojamas MEAN_C
mean_thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY, blockSize=11, C=2)

# Adaptuojamas GAUSSIAN_C
gauss_thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, blockSize=11, C=2)

# Palyginimas
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(mean_thresh, cmap='gray')
plt.title('Adaptuojamas MEAN_C')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gauss_thresh, cmap='gray')
plt.title('Adaptuojamas GAUSSIAN_C')
plt.axis('off')

save_fig(plt.gcf(), "07_adaptyvus_palyginimas.png")

print(f"Visi Lab2 paveikslai išsaugoti aplanke: {OUT_DIR.resolve()}")
