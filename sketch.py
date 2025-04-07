import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from scipy.signal import convolve2d
from utils import verifica_dtype, plot_in_grid, plot_image


def gaussian_kernel(size, sigma):
    """Cria um kernel 2D Gaussiano normalizado"""

    # Vetor de coordenadas centrado em 0: -10...0...10
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)

    # Grid 2d com as coordenadas
    xx, yy = np.meshgrid(ax, ax)

    # Aplica a fórmula da gaussiana
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))

    # Normaliza o kernel (soma de todos os valores dá 1)
    return kernel / np.sum(kernel)


if __name__ == '__main__':
    # Ler a imagem
    watch = io.imread('images/crimson.png')

    # Verifica intervalo de pixels
    print(verifica_dtype(watch))

    # Transformar em escala de cinza
    gray_watch = 0.2126 * watch[:, :, 0] + 0.7152 * watch[:, :, 1] + 0.0722 * watch[:, :, 2]


    # Criando o kernel
    kernel = gaussian_kernel(31, 5)

    # Aplica a convolve2d do scipy para comparação
    blurred_watch = convolve2d(gray_watch, kernel, mode='same', boundary='symm')

    # Realçe dos contornos
    watch_sketch = np.divide(gray_watch, blurred_watch)

    # Evita a perda de contraste
    watch_sketch = np.clip(watch_sketch, 0, 1)

    # Mostra as 4 imagens lado a lado
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(gray_watch, cmap='gray')
    plt.title('Escala de cinza')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(blurred_watch, cmap='gray')
    plt.title('Imagem com desfoque')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(watch_sketch, cmap='gray')
    plt.title('Efeito de Esboço')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
