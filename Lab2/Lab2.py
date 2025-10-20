import cv2
import matplotlib.pyplot as plt

# 1. Įkėlimas pilkumo režimu
gray = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Patikrinimas
plt.figure(figsize=(6, 4))
plt.imshow(gray, cmap='gray')
plt.title('Originalus pilkumo vaizdas')
plt.axis('off')
plt.show()

# Histogramos braižymas
plt.figure(figsize=(6, 4))
plt.hist(gray.ravel(), bins=256, range=(0, 256))
plt.title('Pilkumo histograma')
plt.xlabel('Intensyvumas')
plt.ylabel('Pikselių skaičius')
plt.show()

# 2. Globalus (fiksuotas) slenkstis
# Trys skirtingi slenksčiai
for T in [100, 128, 180]:
    _, binary = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
    plt.figure()
    plt.imshow(binary, cmap='gray')
    plt.title(f'Fiksuotas slenkstis T = {T}')
    plt.axis('off')
    plt.show()

# 3. Automatinis (Otsu) metodas
T_otsu, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Otsu parinktas slenkstis:", T_otsu)

plt.imshow(binary_otsu, cmap='gray')
plt.title(f'Otsu metodas (T = {T_otsu:.0f})')
plt.axis('off')
plt.show()

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

plt.show()
