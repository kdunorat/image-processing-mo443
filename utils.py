import numpy as np
    

def verifica_dtype(img):

    # Verificar dtype
    print("Tipo de dado (dtype):", img.dtype)
    
    return [np.min(img), np.max(img)]


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

