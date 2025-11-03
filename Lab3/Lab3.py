# -*- coding: utf-8 -*-
"""
Tekstūrų analizė: GLCM, LBP ir paprastas klasifikatorius

Instrukcija:
1) Į katalogą ./data/ sudėkite po kelis pilkumo vaizdus kiekvienai tekstūrai.
   Pvz.:
     ./data/medis/wood1.jpg, wood2.jpg
     ./data/akmuo/stone1.jpg, stone2.jpg
     ./data/zole/grass1.jpg, ...
     ./data/smelis/sand1.jpg, ...
2) Paleiskite šį failą. Skriptas:
   - Perskaitys vaizdus
   - Apskaičiuos GLCM požymius keliomis kryptimis (0°, 45°, 90°, 135°) ir keliomis distancijomis
   - (Papildomai) Apskaičiuos LBP ir parodys LBP žemėlapį pirmam vaizdui
   - Sudarys lentelę (pandas.DataFrame) ir atvaizduos palyginimo grafikus
   - Sukurs paprastą KNN klasifikatorių ir įvertins tikslumą (train/test skaidymas)

Pastaba: Jei ./data/ nėra, skriptas bandys įkelti failą 'wood.jpg' tame pačiame kataloge kaip pavyzdį.
"""

import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Kur bus saugomi paveikslai (Lab3)
OUT_DIR = Path("lab3_outputs")
OUT_DIR.mkdir(exist_ok=True)

# Pagalbinė funkcija: išsaugo ir uždaro figūrą (be rodymo)
def save_fig(fig, name):
    out_path = OUT_DIR / name
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Išsaugota: {out_path}")

# ----- Pagalbinės funkcijos -----

def read_gray(path):
    """Nuskaito vaizdą kaip pilkumo (uint8) ir grąžina ndarray."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Nepavyko įkelti vaizdo: {path}")
    # Jei bitų gylis ne 8, suprojektuokime į [0, 255]
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def compute_glcm_features(img, distances=(1, 2), angles_deg=(0, 45, 90, 135), levels=256):
    """Apskaičiuoja GLCM požymius keliomis kryptimis/distancijomis ir grąžina vidurkius.

    Požymiai: kontrastas, energija, homogeniškumas, koreliacija.
    """
    angles = np.deg2rad(angles_deg)
    glcm = graycomatrix(
        img,
        distances=list(distances),
        angles=list(angles),
        levels=levels,
        symmetric=True,
        normed=True,
    )
    feats = {}
    for prop in ("contrast", "energy", "homogeneity", "correlation"):
        vals = graycoprops(glcm, prop)
        feats[prop] = float(np.mean(vals))
    return feats


def compute_lbp_hist(img, P=8, R=1, method="uniform"):
    """LBP žemėlapis ir normalizuota histograma (požymiams)."""
    lbp = local_binary_pattern(img, P=P, R=R, method=method)
    n_bins = P + 2  # 'uniform' atveju
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return lbp.astype(np.float32), hist.astype(np.float32)


def collect_dataset(data_dir="./data"):
    """Surenka (label, path) poras iš ./data/<label>/*.jpg|png|jpeg"""
    if not os.path.isdir(data_dir):
        return []
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    samples = []
    for label in sorted(os.listdir(data_dir)):
        cls_dir = os.path.join(data_dir, label)
        if not os.path.isdir(cls_dir):
            continue
        for pat in patterns:
            for p in glob.glob(os.path.join(cls_dir, pat)):
                samples.append((label, p))
    return samples


# ----- Vizualizacijos -----

def plot_feature_bars(df_means, features=("contrast", "energy", "homogeneity", "correlation")):
    """Nubraižo atskirus stulpelinius grafikus kiekvienam požymiui."""
    for feat in features:
        if feat not in df_means.columns:
            continue
        fig = plt.figure()
        df_means[feat].plot(kind="bar")
        plt.title(f"Požymio '{feat}' vidurkis pagal tekstūrą")
        plt.xlabel("Tekstūra")
        plt.ylabel(feat)
        plt.tight_layout()
        safe_name = feat.replace(" ", "_")
        save_fig(fig, f"01_bars_{safe_name}.png")


def main():
    samples = collect_dataset("./data")

    # Jei nerasta ./data, bandome vieną paveikslą kaip demonstraciją
    if len(samples) == 0:
        demo_path = "wood.jpg"
        if os.path.isfile(demo_path):
            samples = [("medis", demo_path)]
            print("[Įspėjimas] Nerastas ./data/, naudojamas pavyzdinis 'wood.jpg'.")
        else:
            print("[Klaida] Nerasta ./data/ ir failo 'wood.jpg'. Įkelkite vaizdus pagal instrukcijas ir paleiskite dar kartą.")
            return

    records = []
    first_img_for_lbp = None
    first_img_path = None

    for label, path in samples:
        img = read_gray(path)
        if first_img_for_lbp is None:
            first_img_for_lbp = img
            first_img_path = path
        glcm_feats = compute_glcm_features(img)
        rec = {
            "label": label,
            "path": path,
            **glcm_feats,
        }
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    print("\n--- GLCM požymių lentelė ---")
    print(df[["label", "contrast", "energy", "homogeneity", "correlation"]])

    # Grupės vidurkiai pagal tekstūrą (label)
    df_means = (
        df.groupby("label")[
            ["contrast", "energy", "homogeneity", "correlation"]
        ]
        .mean(numeric_only=True)
        .sort_index()
    )

    # Grafikai: kiekvienam požymiui atskiras bar chart
    plot_feature_bars(df_means)

    # (Papildoma) LBP demonstracija pirmam vaizdui
    if first_img_for_lbp is not None:
        lbp_map, lbp_hist = compute_lbp_hist(first_img_for_lbp, P=8, R=1, method="uniform")
        fig_orig = plt.figure()
        plt.imshow(first_img_for_lbp, cmap="gray")
        plt.title(f"Originalus vaizdas: {os.path.basename(first_img_path)}")
        plt.axis("off")
        save_fig(fig_orig, "10_originalus_pirmas.png")

        fig_lbp = plt.figure()
        plt.imshow(lbp_map, cmap="gray")
        plt.title("LBP tekstūros žemėlapis (P=8, R=1, 'uniform')")
        plt.axis("off")
        save_fig(fig_lbp, "11_lbp_zemelapis.png")

        fig_lbp_hist = plt.figure()
        plt.plot(lbp_hist)
        plt.title("LBP histogramą (normalizuota)")
        plt.xlabel("LBP reikšmė")
        plt.ylabel("Dažnis")
        plt.tight_layout()
        save_fig(fig_lbp_hist, "12_lbp_histograma.png")

    # Paprastas klasifikatorius su GLCM požymiais (jei turime >= 2 klasių)
    unique_labels = df["label"].unique()
    class_counts = df["label"].value_counts()
    min_count = int(class_counts.min()) if len(class_counts) > 0 else 0

    if len(unique_labels) >= 2 and len(df) >= len(unique_labels) * 2:
        X = df[["contrast", "energy", "homogeneity", "correlation"]].values
        y = df["label"].values

        # Parinktys: naudok KNN (numatyta) arba SVM
        use_svm = False  # pakeisk į True jei nori SVM (RBF)
        if use_svm:
            base_clf = SVC(kernel="rbf", C=10.0, gamma="scale")
        else:
            # KNN pritaikytas mažoms imtims: n_neighbors=1, svertai pagal atstumą
            base_clf = KNeighborsClassifier(n_neighbors=1, weights="distance")

        clf = make_pipeline(StandardScaler(), base_clf)

        # Jei kiekvienoje klasėje turime bent po 2 pavyzdžius – daryk stratified CV
        if min_count >= 2:
            n_splits = min(5, min_count)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, y, cv=skf)
            y_pred = cross_val_predict(clf, X, y, cv=skf)

            print("\n--- Klasifikatorius (GLCM požymiai, stratified CV) ---")
            print(f"Tikslumas (vidurkis per {n_splits}-fold): {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            print(classification_report(y, y_pred, zero_division=0))
            print("Sumišimo matrica (confusion matrix):")
            print(confusion_matrix(y, y_pred, labels=sorted(unique_labels)))
        else:
            # Atsarginis variantas labai mažoms imtims: paprastas train/test skaidymas
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y if len(df) >= len(unique_labels) * 2 else None
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            print("\n--- Klasifikatorius (GLCM požymiai, hold-out) ---")
            print(f"Tikslumas: {accuracy_score(y_test, y_pred):.3f}")
            print(classification_report(y_test, y_pred, zero_division=0))
            print("Sumišimo matrica (confusion matrix):")
            print(confusion_matrix(y_test, y_pred, labels=sorted(unique_labels)))
    else:
        print("\n[Pastaba] Klasifikatoriui reikia bent 2 klasių ir bent po kelis pavyzdžius kiekvienoje klasėje. Įkelkite daugiau vaizdų į ./data/<klasė>/...")

    # Papildoma: parodyti vieno vaizdo GLCM požymius aiškiai
    example_row = df.iloc[0]
    print(
        f"\nPavyzdžio požymiai ({os.path.basename(example_row['path'])}, klasė '{example_row['label']}'):\n"
        f"  Kontrastas:    {example_row['contrast']:.3f}\n"
        f"  Energija:      {example_row['energy']:.3f}\n"
        f"  Homogeniškumas:{example_row['homogeneity']:.3f}\n"
        f"  Koreliacija:   {example_row['correlation']:.3f}\n"
    )
    print(f"Visi Lab3 paveikslai išsaugoti aplanke: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()