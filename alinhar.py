from skimage import io, img_as_ubyte, color, feature, transform
import cv2
from skimage.io import imsave
from skimage.transform import rotate
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter


def get_profile(image):
    height, width = image.shape
    profile = []
    for i in range(height):
        row_sum = np.sum(image[i, :])
        row_sum = width - row_sum
        profile.append(row_sum)

    return profile


def get_rotation_dict(image, theta_1, theta_2, step):
    angles = range(theta_1, theta_2+step, step)
    diff_sum_dict = {}

    for t in angles:
        rotated_image = rotate(image, angle=t, preserve_range=True)
        # Binariza novamente
        rotated_bin = (rotated_image > 0.5).astype(np.uint8)
        profile = get_profile(rotated_bin)
        diff_sum_dict[t] = objective_function_horizontal(profile)

    return diff_sum_dict


def objective_function_horizontal(profile):
    "Calcula a soma do quadrado das diferenças do perfil das linhas adjacentes"
    diff = np.diff(profile)

    return np.sum(diff**2)


#--------------------------------------- Hough


def hough_accumulator(coords, theta_1, theta_2, step):
    accumulator = Counter()
    degrees  = np.arange(theta_1, theta_2 + step, step)
    thetas = np.deg2rad(degrees)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    for x, y in coords:
        rhos = x * cos_t + y * sin_t
        rho_group = np.round(rhos / 1).astype(int)
        accumulator.update(zip(rho_group, degrees))

    return accumulator

def objective_function_hough(accumulator):
    """Calcula ângulo médio das linhas mais comuns"""
    peaks = accumulator.most_common(20)
    thetas = np.array([theta for (_, theta), _ in peaks], dtype=float)
    votes  = np.array([v for  (_, _),      v in peaks], dtype=float)


    return np.average(thetas, weights=votes)




if __name__ == '__main__':
    original = io.imread('imagens_inclinadas/neg_4.png')
    image = original[:, :, :3]
    aiai = color.rgb2gray(image)
    # Binariza
    image_bin = (color.rgb2gray(image) == 1).astype(np.uint8)

    theta_1, theta_2, step = -90, 90, 0.5

    # horizontal

    # scores = get_rotation_dict(image_bin, theta_1, theta_2, step)
    # best_angle = max(scores, key=scores.get)

    # Colocar para aqui precisar do - também.
    # aligned_image = rotate(original, angle=best_angle)


    # Hough
    # Aplica detector de bordas canny
    image_bin_canny = feature.canny(image_bin.astype(bool), sigma=1.0)
    mask = image_bin_canny == True
    # coordenadas de todos os pixels pretos
    coords = np.column_stack(np.where(mask))
    accumulator = hough_accumulator(coords, theta_1, theta_2, step)
    best_theta = objective_function_hough(accumulator)
    print(f"Ângulo estimado de desalinhamento: {best_theta:.2f}°")
    best_theta = float(best_theta)
    aligned_image = rotate(original, angle=-best_theta)

    plt.figure(figsize=(12, 8))
    plt.imshow(aligned_image, cmap='gray')
    plt.axis('off')
    plt.show()

    # imsave('output_t3/inclinado_4.png', img_as_ubyte(aligned_image))
