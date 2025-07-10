import os
import matplotlib.pyplot as plt

# Basitçe bir klasördeki sınıfların görsel sayılarını sayar
def count_images(path):
    counts = {}
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            imgs = [f for f in os.listdir(cls_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            counts[cls] = len(imgs)
    return counts

# train/val/test dizinleri için bar grafikleri çizer
def plot_distribution(base_dir):
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.isdir(split_dir):
            continue
        counts = count_images(split_dir)

        plt.figure()
        plt.bar(counts.keys(), counts.values())
        plt.title(f"{split.capitalize()} Set – Class Counts")
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
