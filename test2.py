# -*- coding: utf-8 -*-

import cv2
from neural_network import grad_cam
import numpy as np

def evaluate_grad_cam(image_path):
    image = cv2.imread(image_path)
    image_flipped = cv2.flip(image, 1)

    heatmap_original = grad_cam(image)
    heatmap_flipped = grad_cam(image_flipped)

    # Evaluación visual

    cv2.imshow('Original', heatmap_original)
    cv2.imshow('Flipped', heatmap_flipped)
    cv2.waitKey(0)

    # Evaluación cuantitativa

    distance = np.linalg.norm(heatmap_original - heatmap_flipped)
    correlation = np.corrcoef(heatmap_original.flatten(), heatmap_flipped.flatten())[0][1]
    print('Distancia euclidiana:', distance)
    print('Correlación de Pearson:', correlation)


if __name__ == "__main__":
    image_path = 'image.jpg'
    evaluate_grad_cam(image_path)
# -*- coding: utf-8 -*-

def evaluate_grad_cam(image_path):
    image = cv2.imread(image_path)
    image_flipped = cv2.flip(image, 1)

    heatmap_original = grad_cam(image)
    heatmap_flipped = grad_cam(image_flipped)

    # Evaluación visual

    cv2.imshow('Original', heatmap_original)
    cv2.imshow('Flipped', heatmap_flipped)
    cv2.waitKey(0)

    # Evaluación cuantitativa

    distance = np.linalg.norm(heatmap_original - heatmap_flipped)
    correlation = np.corrcoef(heatmap_original.flatten(), heatmap_flipped.flatten())[0][1]
    print('Distancia euclidiana:', distance)
    print('Correlación de Pearson:', correlation)


if __name__ == "__main__":
    image_path = 'image.jpg'
    evaluate_grad_cam(image_path)
