import matplotlib.pyplot as plt
from skimage import io, color
import numpy as np
from scipy.signal import convolve2d


guine = io.imread('images/guine.png')
gray_guine = color.rgb2gray(guine)
# Filters

h1 = np.array([
    [ 0,  0, -1,  0,  0],
    [ 0, -1, -2, -1,  0],
    [-1, -2, 16, -2, -1],
    [ 0, -1, -2, -1,  0],
    [ 0,  0, -1,  0,  0]
])

h2 = (1/256) * np.array([
    [ 1,  4,  6,  4, 1],
    [ 4, 16, 24, 16, 4],
    [ 6, 24, 36, 24, 6],
    [ 4, 16, 24, 16, 4],
    [ 1,  4,  6,  4, 1]
])

h3 = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
])

h4 = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

h5 = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

h6 = (1/9) * np.ones((3, 3))

h7 = np.array([
    [-1, -1,  2],
    [-1,  2, -1],
    [ 2, -1, -1]
])

h8 = np.array([
    [ 2, -1, -1],
    [-1,  2, -1],
    [-1, -1,  2]
])

h9 = (1/9) * np.eye(9)

h10 = (1/8) * np.array([
    [-1, -1, -1, -1, -1],
    [-1,  2,  2,  2, -1],
    [-1,  2,  8,  2, -1],
    [-1,  2,  2,  2, -1],
    [-1, -1, -1, -1, -1]
])

h11 = np.array([
    [-1, -1,  0],
    [-1,  0,  1],
    [ 0,  1,  1]
])

def apply_filter(image, kernel):
    return convolve2d(image, kernel, mode='same', boundary='symm')

# Aplicando os filtros
h1_transformed = apply_filter(gray_guine, h1)
h2_transformed = apply_filter(gray_guine, h2)
h3_transformed = apply_filter(gray_guine, h3)
h4_transformed = apply_filter(gray_guine, h4)
h5_transformed = apply_filter(gray_guine, h5)
h6_transformed = apply_filter(gray_guine, h6)
h7_transformed = apply_filter(gray_guine, h7)
h8_transformed = apply_filter(gray_guine, h8)
h9_transformed = apply_filter(gray_guine, h9)
h10_transformed = apply_filter(gray_guine, h10)
h11_transformed = apply_filter(gray_guine, h11)

# Combinação usando raiz da soma dos quadrados
h3_h4_combined = np.sqrt(h3_transformed**2 + h4_transformed**2)

plt.imshow(h3_h4_combined, cmap='gray')
plt.axis('off')
plt.show()
# # Exibir resultados
# fig, axs = plt.subplots(3, 4, figsize=(16, 10))
# axs = axs.ravel()
# titles = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11']
#
# transformed = [h1_transformed, h2_transformed, h3_transformed, h4_transformed, h5_transformed,
#                h6_transformed, h7_transformed, h8_transformed, h9_transformed, h10_transformed,
#                h11_transformed]
#
# for i in range(11):
#     axs[i].imshow(transformed[i], cmap='gray')
#     axs[i].set_title(titles[i])
#     axs[i].axis('off')
#
# axs[11].axis('off')
#
# plt.tight_layout()
# plt.show()