#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from skimage import io, img_as_ubyte, color, filters
from skimage.io import imsave
from skimage.transform import rotate
from collections import Counter


def parse_cmd():
    p = argparse.ArgumentParser(description="Aplica Desalinhamento")
    p.add_argument("input",  type=Path,   help="imagem de entrada PNG")
    p.add_argument("mode",    type=str,    help="tecnina de alinhamento: hough ou horizontal")
    p.add_argument("output", type=Path,   help="imagem de saída PNG")
    p.add_argument("--step",   "-s",  type=float, default=0.5)

    return p.parse_args()

#--------------------------------------- Horizontal
def get_profile(image):
    height, width = image.shape
    profile = []
    for i in range(height):
        row_sum = np.sum(image[i, :])
        row_sum = width - row_sum
        profile.append(row_sum)

    return profile


def get_rotation_dict(image, theta_1, theta_2, step):
    angles = np.arange(theta_1, theta_2+step, step)
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
    """Calcula uma média ponderada dos angulos das linhas mais comuns"""
    peaks = accumulator.most_common(20)
    thetas = np.array([theta for (_, theta), _ in peaks], dtype=float)
    votes  = np.array([v for  (_, _), v in peaks], dtype=float)


    return np.average(thetas, weights=votes)


def main():
    args = parse_cmd()
    if args.mode.lower() not in ["hough", "horizontal"]:
        raise SystemExit("Modo não reconhecido. Use: hough ou horizontal")

    # leitura
    original = io.imread(args.input)
    gray_image = color.rgb2gray(original[:, :, :3])
    # Binarização
    thr = filters.threshold_otsu(gray_image)
    image_bin = (gray_image < thr).astype(np.uint8)

    theta_1, theta_2, step = -90, 90, args.step

    # Horizontal
    if args.mode == 'horizontal':
        scores = get_rotation_dict(image_bin, theta_1, theta_2, step)
        best_angle = max(scores, key=scores.get)
        best_angle = - best_angle

    # Hough
    else:
        # Aplica um filtro sobel
        edge_sobel = filters.sobel_h(image_bin.astype(float))
        mask = edge_sobel > 0
        # coordenadas de todos os pixels pretos
        coords = np.column_stack(np.where(mask))
        accumulator = hough_accumulator(coords, theta_1, theta_2, step)
        best_angle = objective_function_hough(accumulator)

    print(f"Ângulo estimado de desalinhamento: {best_angle:.2f}°")
    # alinha imagem
    aligned_image = rotate(original, angle=-best_angle, resize=True)
    # salva
    imsave(args.output, img_as_ubyte(aligned_image))


if __name__ == '__main__':
    main()
