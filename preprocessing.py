import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
import cv2

def grayscale_and_resize(image_path, size=(256, 256)):
    """
    görüntüyü gri tonlamaya çevirir, yeniden boyutlandırır ve uint8 formatında döner.
    """
    img = imread(image_path, as_gray=True)
    img_resized = resize(img, size, anti_aliasing=True)
    return (img_resized * 255).astype(np.uint8)


def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def batch_preprocess(input_dir, output_dir, size=(256, 256)):
    """
    input_dir içindeki tüm .png/.jpeg dosyalarını grayscale_and_resize ile işleyip
    output_dir içine aynı adla kaydeder.
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.png', '.jpeg', '.jpg')):
            src_path = os.path.join(input_dir, fname)
            dst_path = os.path.join(output_dir, fname)
            img = grayscale_and_resize(src_path, size)
            img= apply_clahe(img)
            imsave(dst_path, img)
