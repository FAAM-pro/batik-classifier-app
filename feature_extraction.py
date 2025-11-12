import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

def crop_center(gray_img):
    tinggi, lebar = gray_img.shape
    ymin, ymax = tinggi // 3, tinggi * 2 // 3
    xmin, xmax = lebar // 3, lebar * 2 // 3
    return gray_img[ymin:ymax, xmin:xmax]

def extract_glcm_features(img, props, dists=[5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    glcm = graycomatrix(img, distances=dists, angles=angles, levels=256, symmetric=True, normed=True)
    features = []
    for prop in props:
        values = graycoprops(glcm, prop)[0]
        features.extend(values)
    return features

def load_dataset(base_dir="E:\Citra"):
    props = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    features, labels = [], []

    for class_name in os.listdir(os.path.join(base_dir, "batik")):
        folder_path = os.path.join(base_dir, "batik", class_name)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cropped = crop_center(gray)
            resized = cv2.resize(cropped, (128, 128))
            feats = extract_glcm_features(resized, props)
            features.append(feats)
            labels.append(class_name)

    columns = [f"{prop}_{ang}" for prop in props for ang in ['0', '45', '90', '135']]
    df = pd.DataFrame(features, columns=columns)
    df['label'] = labels
    df['label'] = df['label'].map({'jelamprang': 0, 'jlamprang': 1, 'buketan': 2})
    df.to_csv("glcm_train.csv", index=False)
    return df