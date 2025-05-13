import numpy as np
import os
from freq_filter import get_fourier_spectra, reverse_fft
from skimage import io, img_as_ubyte, color
import matplotlib.pyplot as plt


def compress_image_freq(image, threshold):
    """Compressão no domínio da frequência"""
    # Aplica FFT
    img_transformed, magnitude, _ = get_fourier_spectra(image)
    # Pega a magnitude
    magnitude = np.abs(img_transformed)

    # Zera os coeficientes com magnitude menor que o limiar
    img_transformed[magnitude < threshold] = 0

    # Inverso para voltar pro espaço da imagem
    compressed_img = reverse_fft(img_transformed)

    return compressed_img

def save_histogram(data, filename):
    plt.figure()
    plt.hist(data, bins=256, range=(0, 255), color='black')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    output_dir = '../output_t2'
    img_path = '../images/guine.png'
    guine = io.imread(img_path)
    gray_guine = color.rgb2gray(guine)
    threshold = 1000
    compressed_guine = compress_image_freq(gray_guine, threshold=threshold)
    # Normaliza para [0,1]
    compressed_guine = compressed_guine - compressed_guine.min()
    compressed_guine = compressed_guine / compressed_guine.max()

    # Histogramas
    original_histogram = gray_guine.ravel()
    compressed_histogram = compressed_guine.ravel()
    io.imsave(f'{output_dir}/compressed_image_{threshold}.png', img_as_ubyte(compressed_guine))
    # Ajusta a escala para o histograma
    gray_guine = gray_guine * 255
    compressed_guine = compressed_guine * 255
    save_histogram(gray_guine.ravel(), f'{output_dir}/histograma_original.png')
    save_histogram(compressed_guine.ravel(), f'{output_dir}/histograma_comprimido_{threshold}.png')

    # Calcula tamanhos
    original_size_kb = os.path.getsize(img_path) / 1024
    compressed_size_kb = os.path.getsize(f'{output_dir}/compressed_image_{threshold}.png') / 1024

    # Calcula compressão percentual
    compression_ratio = 100 * (1 - compressed_size_kb / original_size_kb)
    # Print final
    print(f"Original size: {original_size_kb:.2f} KB")
    print(f"Compressed image size: {compressed_size_kb:.2f} KB")
    print(f"Compression ratio: {compression_ratio:.2f}%")
