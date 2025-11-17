import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# =========================
# Pagalbinės funkcijos
# =========================

def get_neighbors(i, j, shape, neighborhood=4):
    h, w = shape
    neighbors = []
    directions_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    directions_8 = directions_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    directions = directions_4 if neighborhood == 4 else directions_8

    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < h and 0 <= nj < w:
            neighbors.append((ni, nj))
    return neighbors


def local_energy(i, j, label, labels, img, beta=1.0, neighborhood=4):
    data_term = (label - img[i, j]) ** 2
    smoothness = 0
    for ni, nj in get_neighbors(i, j, labels.shape, neighborhood):
        smoothness += beta * (label != labels[ni, nj])
    return data_term + smoothness


def total_energy(labels, img, beta=1.0, neighborhood=4):
    h, w = labels.shape
    data_term = np.sum((labels - img) ** 2)
    smoothness = 0

    for i in range(h):
        for j in range(w):
            if i + 1 < h and labels[i, j] != labels[i + 1, j]:
                smoothness += beta
            if j + 1 < w and labels[i, j] != labels[i, j + 1]:
                smoothness += beta

            if neighborhood == 8:
                if i + 1 < h and j + 1 < w and labels[i, j] != labels[i + 1, j + 1]:
                    smoothness += beta
                if i + 1 < h and j - 1 >= 0 and labels[i, j] != labels[i + 1, j - 1]:
                    smoothness += beta

    return data_term + smoothness


def icm(img, init_labels, beta=1.0, n_iters=10, neighborhood=4, verbose=True):
    labels = init_labels.copy()
    h, w = labels.shape

    for it in range(n_iters):
        changes = 0
        for i in range(h):
            for j in range(w):
                current_label = labels[i, j]
                energies = [
                    local_energy(i, j, lab, labels, img, beta, neighborhood)
                    for lab in [0, 1]
                ]
                best_label = np.argmin(energies)
                if best_label != current_label:
                    labels[i, j] = best_label
                    changes += 1

        if verbose:
            print(f"Iteracija {it+1}: pakeitimų = {changes}")

        if changes == 0:
            break

    return labels


# =========================
# Pagrindinis kodas
# =========================

if __name__ == "__main__":
    OUTPUT_DIR = "lab6_outputs/"    # ← Keisk čia jei nori
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    IMAGE_PATH = "/Users/ainius/PycharmProjects/VSlabs/Lab4/lab4_outputs/1_coin_original.png"
    THRESH = 0.5
    NEIGHBORHOOD = 4
    MAX_ITERS = 10

    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Nerastas paveikslėlis")

    img = img.astype(np.float32) / 255.0

    # 1. Originalus → į failą
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, "1_original.png"), bbox_inches='tight')
    plt.close()

    # 2. Pradinė segmentacija
    init_labels = (img > THRESH).astype(int)

    plt.imshow(init_labels, cmap="gray")
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, "2_initial_segmentation.png"), bbox_inches='tight')
    plt.close()

    # 3. ICM β=1.0
    beta_main = 1.0
    final_labels_main = icm(img, init_labels, beta_main, MAX_ITERS, NEIGHBORHOOD)

    plt.imshow(final_labels_main, cmap="gray")
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, "3_final_segmentation_beta1.png"), bbox_inches='tight')
    plt.close()

    # ---- 4. Eksperimentai su skirtingais beta ----
    beta_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1]
    print("\n=== Lentelė: beta reikšmės ir galutinė energija ===")
    print("beta\tEnergija")

    results = []
    segmentation_results = []  # čia susirinksime (beta, labels) poras subplot'ui

    for beta in beta_values:
        print(f"\n--- ICM su beta = {beta} ---")
        labels_beta = icm(
            img,
            init_labels,
            beta=beta,
            n_iters=MAX_ITERS,
            neighborhood=NEIGHBORHOOD,
            verbose=False  # čia tyliu režimu, be "Iteracija ..."
        )

        E_final = total_energy(labels_beta, img, beta=beta, neighborhood=NEIGHBORHOOD)
        results.append((beta, E_final))
        segmentation_results.append((beta, labels_beta))
        print(f"beta = {beta:.2f}, galutinė energija = {E_final:.4f}")

    # Išsaugom lentelę su energijomis
    with open(os.path.join(OUTPUT_DIR, "beta_results.txt"), "w") as f:
        f.write("beta\tenergy\n")
        for beta, E in results:
            f.write(f"{beta}\t{E:.4f}\n")

    # ---- 5. Viena didelė nuotrauka su visais beta ----
    n = len(beta_values)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))

    if n == 1:
        axes = [axes]  # jeigu būtų tik vienas beta, kad kodas vis tiek veiktų

    for idx, (beta, labels_beta) in enumerate(segmentation_results):
        ax = axes[idx]
        ax.imshow(labels_beta, cmap='gray')
        ax.set_title(f"β = {beta}")
        ax.axis('off')

    betas_str = ", ".join(str(b) for b in beta_values)
    fig.suptitle(f"Segmentacija su skirtingomis β reikšmėmis: {betas_str}", fontsize=12)

    plt.tight_layout()
    # kad suptitle neprispaustų viršaus
    plt.subplots_adjust(top=0.8)

    out_path = os.path.join(OUTPUT_DIR, "beta_comparison.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    print("Bendra beta palyginimo nuotrauka išsaugota:", out_path)
