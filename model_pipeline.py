from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

# Esnek pipeline, PCA ve scaler kullanılabilir ya da atlanabilir

def build_pipeline(model: BaseEstimator, use_pca: bool = True, use_scaler: bool = True, n_components: float = 0.95) -> Pipeline:
    steps = []
    if use_scaler:
        steps.append(('scaler', StandardScaler()))
    if use_pca:
        steps.append(('pca', PCA(n_components=n_components, svd_solver='auto')))
    steps.append(('clf', model))
    return Pipeline(steps)
