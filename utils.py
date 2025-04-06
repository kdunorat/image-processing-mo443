import numpy as np
import matplotlib.pyplot as plt


def verifica_dtype(img):
    """Printa o data type da imagem e retorna o valor mínimo e o máximo de intensidade"""
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

def plot_image(image, title, color = False):
    cmap = None # Define cmap como None por padrão
    if not color:
        cmap = 'gray'
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()




