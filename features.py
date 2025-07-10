from skimage.io import imread
from skimage.feature import hog, local_binary_pattern
import mahotas
import numpy as np
import os

# Her resimden öznitelik çıkaran fonksiyonlar
def extract_hog(img):
    return hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=9, block_norm='L2-Hys')

def extract_lbp(img):
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10), density=True)
    return hist

def extract_haralick(img):
    img_uint8 = (img * 255).astype(np.uint8)
    return mahotas.features.haralick(img_uint8).mean(axis=0)

# Bir klasördeki tüm görüntülerden öznitelik çıkar
def extract_features_from_folder(folder_path, method='hog'):
    X = []
    y = []

    for label in sorted(os.listdir(folder_path)):
        class_path = os.path.join(folder_path, label)
        if not os.path.isdir(class_path): continue

        for fname in os.listdir(class_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')): continue

            img_path = os.path.join(class_path, fname)
            img = imread(img_path)  # zaten grayscale ve 256x256 olarak varsayılıyor

            if method == 'hog':
                features = extract_hog(img)
            elif method == 'lbp':
                features = extract_lbp(img)
            elif method == 'haralick':
                features = extract_haralick(img)
            else:
                raise ValueError("method must be 'hog', 'lbp', or 'haralick'")

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)
