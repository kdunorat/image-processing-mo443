#!/usr/bin/env python3
import numpy as np
import argparse
from skimage import io, img_as_ubyte
from math import floor, sin, cos, radians
import matplotlib.pyplot as plt
import time

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
    y_raw = np.clip(y_raw, 0, h - 1.000001)
    x_raw = np.clip(x_raw, 0, w - 1.000001)
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

    scaled_image = np.empty((round(h * n), round(w * n)), dtype=np.float32)

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


def rotate_image(img, angle_deg, method='bilinear'):
    """Aplica uma rotação na imagem usando mapeamento inverso e interpolação."""

    h_orig, w_orig = img.shape[:2]
    cx_orig, cy_orig = w_orig // 2, h_orig // 2


    angle_rad = radians(angle_deg)
    cos_theta = cos(angle_rad)
    sin_theta = sin(angle_rad)


    # Rotaciona os 4 cantos da imagem original para encontrar a nova dimensão
    cantos = np.array([
        [0, 0],
        [w_orig, 0],
        [w_orig, h_orig],
        [0, h_orig]
    ])
    # Matriz de rotação
    matrix_rot = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])

    cantos_rot = np.dot(cantos - np.array([cx_orig, cy_orig]), matrix_rot.T) + np.array([cx_orig, cy_orig])

    x_coords, y_coords = cantos_rot[:, 0], cantos_rot[:, 1]
    w_novo = int(np.ceil(max(x_coords) - min(x_coords)))
    h_novo = int(np.ceil(max(y_coords) - min(y_coords)))

    # Cria a imagem de saída e seu centro
    img_rotacionada = np.zeros((h_novo, w_novo), dtype=img.dtype)
    cx_novo, cy_novo = w_novo // 2, h_novo // 2

    # Mapeamento Inverso
    for y_out in range(h_novo):
        for x_out in range(w_novo):
            # Translada para o centro da nova imagem
            x_temp = x_out - cx_novo
            y_temp = y_out - cy_novo

            # Aplica rotação inversa
            x_rot = x_temp * cos_theta + y_temp * sin_theta
            y_rot = -x_temp * sin_theta + y_temp * cos_theta

            # Transladar de volta para o sistema da imagem original
            x_in = x_rot + cx_orig
            y_in = y_rot + cy_orig

            # Interpolação
            # Verifica se cabe na imagem original
            if 0 <= x_in < w_orig and 0 <= y_in < h_orig:
                pixel_value = 0
                if method == 'nn':
                    y_nn, x_nn = nn_interpolation([y_in, x_in])
                    if (0 <= y_nn < h_orig and 0 <= x_nn < w_orig):
                        pixel_value = img[y_nn, x_nn]
                else:
                    if method == 'bilinear':
                        pixel_value = bilinear_interpolation(img, (y_in, x_in))
                    elif method == 'bicubic':
                        pixel_value = bicubic_interpolation(img, (y_in, x_in))
                    elif method == 'lagrange':
                        pixel_value = lagrange_interpolation(img, (y_in, x_in))


                img_rotacionada[y_out, x_out] = np.clip(pixel_value, 0, 255)

    return img_rotacionada.astype(img.dtype)


def parse_args():
    parser = argparse.ArgumentParser(description='Realiza transformações geométricas em imagens.')
    parser.add_argument('-i', '--input', required=True, help='Caminho para a imagem de entrada PNG.')
    parser.add_argument('-o', '--output', required=True, help='Caminho para a imagem de saída PNG.')


    transform_group = parser.add_mutually_exclusive_group(required=True)
    transform_group.add_argument('-a', '--angle', type=float, help='Ângulo de rotação em graus.')
    transform_group.add_argument('-e', '--scale', type=float, help='Fator de escala.')

    parser.add_argument('-m', '--method', default='bilinear', choices=['nn', 'bilinear', 'bicubic', 'lagrange'],
                        help='Método de interpolação. Padrão: bilinear.')
    parser.add_argument('-d', '--dims', nargs=2, type=int,
                        help='(Ignorado) Dimensões da imagem de saída (largura altura).')

    return parser.parse_args()

def main():
    args = parse_args()
    try:
        print(f"Lendo a imagem de: {args.input}")
        img_original = io.imread(args.input)
    except FileNotFoundError:
        print(f"Erro: Arquivo de entrada não encontrado em '{args.input}'")
        return

    # Processamento por canal para imagens RGB
    def process_image(img_channel, args):
        if args.scale is not None:
            return scale_image(img_channel, args.scale, args.method)
        elif args.angle is not None:
            return rotate_image(img_channel, args.angle, args.method)
        return None

    if img_original.ndim == 3 and img_original.shape[2] in [3, 4]:  # Suporte para RGB e RGBA
        print(f"Processando imagem RGB com interpolação '{args.method}'...")
        r_channel, g_channel, b_channel = [img_original[:, :, i] for i in range(3)]

        r_out = process_image(r_channel, args)
        g_out = process_image(g_channel, args)
        b_out = process_image(b_channel, args)

        # A pilha de canais precisa ter as mesmas dimensões.
        # Nossas funções já garantem isso.
        output_image = np.stack([r_out, g_out, b_out], axis=-1)

    else:
        print(f"Processando imagem em escala de cinza com interpolação '{args.method}'...")
        output_image = process_image(img_original, args)

    io.imsave(args.output, output_image)

def generate_cropped_baboon(baboon):
    original_img = io.imread('./images/baboon_colorida.png').astype(np.uint8)
    cropped_baboon = original_img[:100, 120:250, :]
    plt.figure()
    plt.imshow(cropped_baboon)
    plt.axis('off')
    plt.savefig('./output_t4/cropped_img.png', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    start_time = time.time()
    main()
    finish_time = time.time()
    finish_time = finish_time - start_time
    print(f"Tempo decorrido: {finish_time:.2f}s")
