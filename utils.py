import numpy as np
import matplotlib.pyplot as plt


def verifica_dtype(img):
    """Printa o data type da imagem e retorna o valor mínimo e o máximo de intensidade"""
    # Verificar dtype
    print("Tipo de dado (dtype):", img.dtype)
    
    return [np.min(img), np.max(img)]

def plot_image(image, title = '', color = False):
    """Plota imagens individuais"""
    cmap = None # Define cmap como None por padrão
    if not color:
        cmap = 'gray'
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def get_out_path(img_name: str):
    return f'output/{img_name}.png'
