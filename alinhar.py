from skimage import io, img_as_ubyte, color
from skimage.io import imsave
from skimage.transform import rotate
import matplotlib.pyplot as plt
import numpy as np


def get_profile(image):
    height, width = image.shape
    profile = []
    for i in range(height):
        row_sum = np.sum(image[i, :])
        row_sum = width - row_sum
        profile.append(row_sum)

    return profile

def get_rotation_dict(image, theta_1, theta_2, step):
    degrees = range(theta_1, theta_2, step)
    diff_sum_dict = {}

    for t in degrees:
        rotated_image = rotate(image, angle=t, preserve_range=True)
        rotated_bin = (rotated_image > 0.5).astype(np.uint8)
        profile = get_profile(rotated_bin)
        diff_sum_dict[t] = objective_function(profile)

    return diff_sum_dict


def objective_function(profile):
    diff = np.diff(profile)

    return np.sum(diff**2)


if __name__ == '__main__':
    image = io.imread('imagens_inclinadas/neg_4.png')[:, :, :3]
    # Binariza
    image = (color.rgb2gray(image) == 1).astype(np.uint8)

    theta_1, theta_2, step = -10, 10, 1
    scores = get_rotation_dict(image, theta_1, theta_2, step)
    best_angle = max(scores, key=scores.get)
    align_image = rotate(image, angle=best_angle, preserve_range=True)


    plt.figure(figsize=(12, 8))
    plt.imshow(align_image, cmap='gray')
    plt.axis('off')
    plt.show()

