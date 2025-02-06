import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load a noisy image
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_3/bunga.jpg", cv2.IMREAD_GRAYSCALE)  # Ensure the image has noise
if image is None:
    print("Error: Could not load image. Please check the file path.")
else:
    # Display original noisy image
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Noisy Image")
    plt.axis('off')
    
    # 1. Noise Reduction Filters

    # Apply Mean filter
    mean_filtered = cv2.blur(image, (5, 5))
    plt.subplot(3, 2, 2)
    plt.imshow(mean_filtered, cmap='gray')
    plt.title("Mean Filtered Image")
    plt.axis('off')

    # Apply Median filter
    median_filtered = cv2.medianBlur(image, 5)
    plt.subplot(3, 2, 3)
    plt.imshow(median_filtered, cmap='gray')
    plt.title("Median Filtered Image")
    plt.axis('off')

    # Apply Gaussian filter
    gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)
    plt.subplot(3, 2, 4)
    plt.imshow(gaussian_filtered, cmap='gray')
    plt.title("Gaussian Filtered Image")
    plt.axis('off')

    # 2. Histogram Equalization for Contrast Enhancement
    equalized_image = cv2.equalizeHist(image)
    plt.subplot(3, 2, 5)
    plt.imshow(equalized_image, cmap='gray')
    plt.title("Histogram Equalized Image")
    plt.axis('off')

    # 3. Geometric Transformations: Rotation and Scaling

    # Define rotation matrix
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    angle = 45  # Rotate by 45 degrees
    scale = 1.2  # Scale by 120%
    
    # Get rotation matrix and perform rotation and scaling
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_scaled_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    plt.subplot(3, 2, 6)
    plt.imshow(rotated_scaled_image, cmap='gray')
    plt.title("Rotated & Scaled Image")
    plt.axis('off')

    # Display all images
    plt.tight_layout()
    plt.show()