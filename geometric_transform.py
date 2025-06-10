import numpy as np
from skimage import io, img_as_ubyte
from skimage.util import img_as_float32
from skimage.io import imsave
import matplotlib.pyplot as plt
from math import floor


def nn_interpolation(p_i):
    """Aplica a interpolação por vizinhos mais próximos"""
    y_raw, x_raw = p_i
    y = floor(y_raw)
    x = floor(x_raw)
    dx = x_raw - x
    dy = y_raw - y

    if dx < 0.5 and dy < 0.5:
        return y, x
    if dx >= 0.5 > dy:
        return y, x + 1
    if dx < 0.5 <= dy:
        return y + 1, x
    else:
        return y + 1, x + 1

def bilinear_interpolation(img, p_i):
    """Calcula o valor do pixel por interpolação bilinear."""
    h, w = img.shape[:2]
    y_raw, x_raw = p_i
    x = floor(x_raw)
    y = floor(y_raw)
    dx = x_raw - x
    dy = y_raw - y

    x1 = min(x + 1, w - 1)
    y1 = min(y + 1, h - 1)
    f_xy = img[y, x]
    f_x1y = img[y, x1]
    f_xy1 = img[y1, x]
    f_x1y1 = img[y1, x1]

    pixel_value = (1 - dx) * (1 - dy) * f_xy + \
                  dx * (1 - dy) * f_x1y + \
                  (1 - dx) * dy * f_xy1 + \
                  dx * dy * f_x1y1

    return pixel_value


def bicubic_interpolation(img, p_i):
    """Aplica a interpolação bicúbica"""
    h, w = img.shape[:2]
    y_raw, x_raw = p_i
    x = floor(x_raw)
    y = floor(y_raw)
    dx = x_raw - x
    dy = y_raw - y
    total = 0

    def P(t):
        return t if t > 0 else 0
    def R(s):
        return (P(s + 2) ** 3 - 4 * P(s + 1) ** 3 + 6 * P(s) ** 3 - 4 * P(s - 1) ** 3) / 6

    for n in range(-1, 3):
        for m in range(-1, 3):
            sample_x = max(0, min(x + m, w - 1))
            sample_y = max(0, min(y + n, h - 1))
            f_val = img[sample_y, sample_x]
            weight = R(m - dx) * R(dy - n)

            total += f_val * weight

    return total

def lagrange_interpolation(img, p_i):
    """Aplica a interpolação por polinomios  de Lagrange."""
    h, w = img.shape[:2]
    y_raw, x_raw = p_i
    x = floor(x_raw)
    y = floor(y_raw)
    dx = x_raw - x
    dy = y_raw - y

    def L(n):
        y_n = y + n - 2


        x_m1 = max(0, min(x - 1, w - 1))
        x_0 = max(0, min(x, w - 1))
        x_1 = max(0, min(x + 1, w - 1))
        x_2 = max(0, min(x + 2, w - 1))

        y_n_clamped = max(0, min(y_n, h - 1))
        f1 = img[y_n_clamped, x_m1]
        f2 = img[y_n_clamped, x_0]
        f3 = img[y_n_clamped, x_1]
        f4 = img[y_n_clamped, x_2]
        part1 = (-dx * (dx - 1) * (dx - 2) / 6) * f1
        part2 = ((dx + 1) * (dx - 1) * (dx - 2) / 2) * f2
        part3 = (-dx * (dx + 1) * (dx - 2) / 2) * f3
        part4 = (dx * (dx + 1) * (dx - 1) / 6) * f4

        return part1 + part2 + part3 + part4

    L1 = L(1)
    L2 = L(2)
    L3 = L(3)
    L4 = L(4)

    part1 = (-dy * (dy - 1) * (dy - 2) / 6) * L1
    part2 = ((dy + 1) * (dy - 1) * (dy - 2) / 2) * L2
    part3 = (-dy * (dy + 1) * (dy - 2) / 2) * L3
    part4 = (dy * (dy + 1) * (dy - 1) / 6) * L4

    return part1 + part2 + part3 + part4


def scale_image(img, n, method='bilinear'):
    """Aplica uma escala na imagem, tratando os diferentes métodos de interpolação."""
    w = img.shape[1]
    h = img.shape[0]

    scaled_image = np.empty((round(img.shape[1] * n), round(img.shape[0] * n)), dtype=np.float32)

    # Matriz de escala (inversa)
    inv_scale_matrix = np.array([[1 / n, 0],
                                 [0, 1 / n]])

    y_o, x_o = scaled_image.shape
    for xo_coord in range(x_o):
        for yo_coord in range(y_o):
            out_coords = np.array([xo_coord, yo_coord])
            in_coords = inv_scale_matrix @ out_coords
            y_in, x_in = in_coords[1], in_coords[0]

            pixel_value = 0

            if method == 'nn':
                # Aqui retorna coordenadas
                scaled_y_coord, scaled_x_coord = nn_interpolation([y_in, x_in])
                if (0 <= scaled_y_coord < h and 0 <= scaled_x_coord < w):
                    pixel_value = img[scaled_y_coord, scaled_x_coord]
            else:
                # esses retornam intensidade
                if method == 'bilinear':
                    pixel_value = bilinear_interpolation(img, (y_in, x_in))
                elif method == 'bicubic':
                    pixel_value = bicubic_interpolation(img, (y_in, x_in))
                elif method == 'lagrange':
                    pixel_value = lagrange_interpolation(img, (y_in, x_in))

            scaled_image[yo_coord, xo_coord] = np.clip(pixel_value, 0, 255)

    return scaled_image.astype(np.uint8)



if __name__ == '__main__':
    original_img = io.imread('./images/baboon_colorida.png').astype(np.uint8)
    cropped_baboon = original_img[:100, 120:250, :]

    n = 5.75
    method = 'nn'
    if cropped_baboon.ndim == 3 and cropped_baboon.shape[2] == 3:
        print("Processando imagem RGB...")
        r_channel = cropped_baboon[:, :, 0]
        g_channel = cropped_baboon[:, :, 1]
        b_channel = cropped_baboon[:, :, 2]

        scaled_r = scale_image(r_channel, n, method)
        scaled_g = scale_image(g_channel, n, method)
        scaled_b = scale_image(b_channel, n, method)

        scaled_image = np.stack([scaled_r, scaled_g, scaled_b], axis=-1)
    else:
        scaled_image = scale_image(cropped_baboon, n=n, method='bilinear')

    plt.imshow(scaled_image)
    plt.axis('off')
    plt.show()


