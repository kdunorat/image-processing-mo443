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

def plot_in_grid(images, titles=None, n_columns=3, size=(12, 8), color = False):
    """Plota imagens em um grid de n colunas"""
    cmap = None  # Define cmap como None por padrão
    if not color:
        cmap = 'gray'
    n = len(images)
    n_linhas = (n + n_columns - 1) // n_columns  # arredonda pra cima

    fig, axs = plt.subplots(n_linhas, n_columns, figsize=size)
    axs = axs.ravel()  # transforma em lista

    for i in range(n_linhas * n_columns):
        if i < n:
            axs[i].imshow(images[i], cmap=cmap)
            if titles:
                axs[i].set_title(titles[i])
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()



