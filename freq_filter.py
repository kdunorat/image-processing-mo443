import os
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from skimage import io, img_as_ubyte

def get_fourier_spectra(image, log = True, shift = True):

    # Aplica a FFT2D
    fft2_transform = fft2(image)

    if shift:
        # Faz o shift para F(0,0) ficar no centro
        fft2_transform = fftshift(fft2_transform)

    # Calcula magnitude e fase
    magnitude = np.abs(fft2_transform)
    phase = np.angle(fft2_transform)

    if log:
        # Aplica o log para possibilitar a visualização
        magnitude = np.log(1 + magnitude)

    return fft2_transform, magnitude, phase,


def reverse_fft(spec):
    """FFT 2-D reversa da centralização (retorna parte real)."""
    return np.real(ifft2(ifftshift(spec)))


def apply_mask(image, mask):
    f_transformed, _, _ = get_fourier_spectra(image, log=False)
    # Aplica a máscara (multiplicação ponto a ponto)
    f_filtered = f_transformed * mask
    # Faz a IFFT e retorna a imagem
    img_filtered = reverse_fft(f_filtered)
    return img_filtered


def create_mask(shape, type_filter, d0, d1=None):
    """Cria mascara para filtrar no domínio da frequencia"""
    height, width = shape
    u = np.arange(height)
    v = np.arange(width)
    # centraliza as menores frequencias
    u = u - height // 2
    v = v - width // 2
    # transforma os vetores 1D em v
    u_2d, v_2d = np.meshgrid(v, u)
    D = np.sqrt(u_2d ** 2 + v_2d ** 2)

    # Cria as mascaras ideais e as mascaras gaussianas
    if type_filter == 'low':
        ideal_mask = D <= d0
        gaussian_mask = np.exp(-(D ** 2) / (2 * (d0 ** 2)))
    elif type_filter == 'high':
        ideal_mask = D >= d0
        gaussian_mask = 1 - np.exp(-(D ** 2) / (2 * (d0 ** 2)))
    elif type_filter == 'bandpass':
        ideal_mask = (D >= d0) & (D <= d1)
        gaussian_mask = np.exp(-(D ** 2) / (2 * (d1 ** 2))) - np.exp(-(D ** 2) / (2 * (d0 ** 2)))
    elif type_filter == 'bandreject':
        ideal_mask = (D <= d0) | (D >= d1)
        gaussian_mask = 1 - (np.exp(-(D ** 2) / (2 * (d1 ** 2))) - np.exp(-(D ** 2) / (2 * (d0 ** 2))))
    else:
        raise ValueError("Tipo de filtro inválido")

    return ideal_mask.astype(float), gaussian_mask


if __name__ == '__main__':
    output_dir = './output_t2'
    baboon_mono = io.imread('./images/baboon_monocromatica.png')

    # Parametros dos filtros
    d0 = 40
    d1 = 80

    masks_ideal = {}
    masks_gaussian = {}
    # Cria dicionários para armazenar as máscaras ideias e as máscaras gaussianas
    for mask_type in ['low', 'high', 'bandpass', 'bandreject']:
        ideal_mask, gaussian_mask = create_mask(baboon_mono.shape, mask_type, d0, d1)

        if mask_type == 'low':
            name = 'Passa-Baixa'
        elif mask_type == 'high':
            name = 'Passa-Alta'
        elif mask_type == 'bandpass':
            name = 'Passa-Faixa'
        elif mask_type == 'bandreject':
            name = 'Rejeita-Faixa'

        masks_ideal[name] = ideal_mask
        masks_gaussian[name] = gaussian_mask


    # Aplica cada filtro e salva o resultado
    for name in masks_ideal.keys():
        img_filtered_ideal = apply_mask(baboon_mono, masks_ideal[name])
        img_filtered_gaussian = apply_mask(baboon_mono, masks_gaussian[name])
        # normaliza a imagem
        img_filtered_ideal = img_filtered_ideal / 255
        img_filtered_gaussian = img_filtered_gaussian / 255

        _, mi, _ = get_fourier_spectra(img_filtered_ideal)
        _, mg, _ = get_fourier_spectra(img_filtered_gaussian)
        # Normaliza os espectros
        mi = mi - mi.min()
        mi = mi / mi.max()
        mg = mg - mg.min()
        mg = mg / mg.max()

        save_path_ideal = os.path.join(output_dir, f'ideal_{name.replace(" ", "_").lower()}.png')
        save_path_ideal_core = os.path.join(output_dir, f'ideal_core_{name.replace(" ", "_").lower()}.png')
        save_path_gaussian = os.path.join(output_dir, f'gaussian_{name.replace(" ", "_").lower()}.png')
        save_path_gaussian_core = os.path.join(output_dir, f'gaussian_core_{name.replace(" ", "_").lower()}.png')

        io.imsave(save_path_ideal, img_as_ubyte(img_filtered_ideal))
        io.imsave(save_path_ideal_core, img_as_ubyte(mi))
        io.imsave(save_path_gaussian, img_as_ubyte(img_filtered_gaussian))
        io.imsave(save_path_gaussian_core, img_as_ubyte(mg))
