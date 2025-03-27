import numpy as np
from skimage import io


def bits_por_pixel(img):

    # Tipo de dado por canal (ex: uint8 = 8 bits)
    bits_per_channel = img.dtype.itemsize * 8

    # Número de canais (ex: 1 para cinza, 3 para RGB)
    num_channels = 1 if img.ndim == 2 else img.shape[2]

    # Bits por pixel
    bpp = bits_per_channel * num_channels

    print(f'Bits por canal: {bits_per_channel}')
    print(f'Número de canais: {num_channels}')
    print(f'Bits por pixel: {bpp}')
    

def verifica_dtype(img):

    # Verificar dtype
    print("Tipo de dado (dtype):", img.dtype)

    # Verificar valores mínimo e máximo
    print("Valor mínimo:", np.min(img))
    print("Valor máximo:", np.max(img))


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

