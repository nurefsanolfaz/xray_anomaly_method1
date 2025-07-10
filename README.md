XRAY_ANOMALY_METHOD1
====================

Bu proje, göğüs röntgeni görüntülerinde anomali tespiti için klasik makine öğrenmesi (ML) ve öznitelik çıkarım yöntemlerine dayalıdır. İki veri kümesi üzerinde çalışılmıştır: Dataset1 (ikili sınıflandırma: Normal vs Pneumonia) ve Dataset2 (beş sınıflı sınıflandırma). 


Bu proje kapsamında kullanılan yöntem: El yapımı öznitelik çıkarımı  (HOG, LBP, HARALICK) + klasik ML (SVM, KNN, MLP, XGBoost)

------------------------------
Klasör Yapısı ve Açıklamaları:
------------------------------

data/ (Görüntü verisi çok fazla yer kapladığı için her set)
├── raw/              -> Orijinal görüntü verisi
├── processed/        -> Yeniden boyutlandırılmış, normalize edilmiş görüntüler (isteğe bağlı)

features/
├── dataset1/         -> Dataset1 için çıkarılmış öznitelikler (HOG, LBP, Haralick, combined)
├── dataset2/         -> Dataset2 için çıkarılmış öznitelikler

models/
├── dataset1/         -> Dataset1 üzerinde eğitilmiş modeller (pipeline formatında .pkl)
├── dataset2/         -> Dataset2 üzerinde eğitilmiş modeller

notebooks/
├── dataset1_training/
│   ├── data_preprocessing.ipynb        -> Dataset1 için görsel işleme ve grayscale + resize
│   ├── feature_extraction.ipynb        -> HOG, LBP, Haralick özniteliklerinin çıkarımı
│   ├── model_training.ipynb            -> SVM/KNN gibi modellerin eğitimi
│   └── model_training_pca_gridsearch.ipynb -> PCA + hiperparametre ayarlama
├── dataset2_training/
│   └── (boş veya eklenebilir)
├── feature_n.ipynb                     -> Dataset2 için ek öznitelik kombinasyon denemeleri
├── test_dataset1.ipynb                 -> Dataset1 modelleriyle test sonuçları
├── test_dataset2.ipynb                 -> Dataset2 modelleriyle test sonuçları

src/method1/
├── __init__.py
├── evaluate.py                         -> Model değerlendirme fonksiyonları (accuracy, classification_report)
├── features.py                         -> Öznitelik çıkarım fonksiyonları (HOG, LBP, Haralick)
├── model_pipeline.py                   -> Pipeline oluşturma ve kayıt işlemleri
├── pca_and_normalization.py            -> PCA + normalization işlemleri
├── preprocessing.py                    -> Görüntü ön işleme (resize, grayscale, dengeleme vb.)
├── visualization_utils.py              -> Confusion matrix ve grafik görselleştirme fonksiyonları

requirements.txt                        -> Gerekli Python kütüphaneleri listesi
README.md                               -> Proje özeti (Markdown)
.gitignore                              -> Takip edilmeyecek dosyalar (örn. __pycache__, .npy, .pkl)

------------------------------
Kullanım:
------------------------------
1. Ortamı kurun:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
2.	Not defterlerini (notebooks/) sırasıyla çalıştırarak:
	•	Görüntüleri ön işleyin
	•	Öznitelikleri çıkarın
	•	Modelleri eğitin ve test edin
3.	Eğitimli modeller models/ altında kayıtlıdır, test defterlerinde yeniden yüklenip kullanılabilir.
4.	Test setleri .npy olarak features/ içinde kayıtlıdır, test defterleri bunları yükleyerek çalışır.
