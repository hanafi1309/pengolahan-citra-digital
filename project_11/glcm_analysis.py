import cv2
import numpy as np 
from skimage.feature import graycomatrix, graycoprops # type: ignore

# Read the grayscale image
image = cv2.imread("c:/Users/Thinkpad/Documents/TUGAS/pengolahan citra digital/project_11/punk.jpg", cv2.IMREAD_GRAYSCALE)

# Calculate GLCM (Gray Level Co-occurrence Matrix)
glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

# Extract texture features
contrast = graycoprops(glcm, 'contrast')[0, 0]
energy = graycoprops(glcm, 'energy')[0, 0]
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
correlation = graycoprops(glcm, 'correlation')[0, 0]

# Display the results
print(f'Contrast: {contrast}')
print(f'Energy: {energy}')
print(f'Homogeneity: {homogeneity}')
print(f'Correlation: {correlation}')
