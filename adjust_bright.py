from skimage import io, color
from skimage.io import imsave
from skimage import img_as_ubyte
import numpy as np
from utils import get_out_path, verifica_dtype


spider = io.imread('images/spider.png')

spider_mono = color.rgb2gray(spider)


def gamma_correction(image, gamma):
    # Normaliza para [0, 1]
    if np.max(image) > 1.0:
        image_norm = np.clip(image / 255.0, 0, 1)
    else:
        image_norm = image

    # Aplica a fórmula à imagem
    corrected = np.power(image_norm, 1 / gamma)

    # Retorna os pixels ao intervalo [0, 255]
    return (corrected * 255).astype(np.uint8)

if __name__ == '__main__':
    # Salva imagem original
    imsave(get_out_path('spider_original'), img_as_ubyte(spider_mono))

    # Testar diferentes gamas e salvar cada uma
    gammas = [1.5, 2.5, 3.5]
    for g in gammas:
        corrected = gamma_correction(spider_mono, g)
        imsave(get_out_path(f'spider_gamma_{g}'), corrected)




