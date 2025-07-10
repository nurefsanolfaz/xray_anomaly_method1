from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple

# PCA fit et ve eğitim verisini dönüştür

def fit_pca(X_train: np.ndarray, n_components: float = 0.95) -> Tuple[PCA, np.ndarray]:
    pca = PCA(n_components=n_components, svd_solver='auto')
    return pca, pca.fit_transform(X_train)

# PCA uygulanmış modeli başka veri için kullan

def transform_pca(pca: PCA, X: np.ndarray) -> np.ndarray:
    return pca.transform(X)

# StandardScaler fit et ve eğitim verisini ölçekle

def fit_scaler(X_train: np.ndarray) -> Tuple[StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    return scaler, scaler.fit_transform(X_train)

# Eğitimde öğrenilen scaler'ı başka veri için kullan

def transform_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X)
