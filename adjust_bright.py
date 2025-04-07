import matplotlib.pyplot as plt
from skimage import io, color
import numpy as np

# Ler a imagem
spider = io.imread('images/spider.png')
spider_mono = color.rgb2gray(spider)

def gamma_correction(image, gamma):
    # Normaliza para [0, 1]
    image_norm = np.clip(image / 255.0, 0, 1)

    # Aplica a fórmula à imagem
    corrected = np.power(image_norm, 1 / gamma)

    # Retorna os pixels ao intervalo [0, 255]
    return (corrected * 255).astype(np.uint8)

if __name__ == '__main__':
    # Testar diferentes gamas
    gammas = [1.5, 2.5, 3.5]
    results = [gamma_correction(spider_mono, g) for g in gammas]


    plt.figure(figsize=(16, 8))
    plt.subplot(1, 4, 1)
    plt.imshow(spider_mono, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # Plota as imagens lado a lado
    for i, (g, img_corr) in enumerate(zip(gammas, results)):
        plt.subplot(1, 4, i+2)
        plt.imshow(img_corr, cmap='gray')
        plt.title(f'Gamma = {g}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()



