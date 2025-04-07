from skimage import io, img_as_ubyte
from skimage.io import imsave
import numpy as np
from utils import get_out_path


pele = io.imread('images/1970_copa_pele_abraco.png')

old_filter = np.array([[0.393, 0.769, 0.189],[0.349, 0.686, 0.168],[0.272, 0.534, 0.131]])

def apply_linear_transform(img, t_filter, bright = 1):
    # Escala
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img / 255.0
    # Aplica um reshape para ficar cada linha um pixel nas 3 colunas (R, G, B)
    flat_img = img.reshape(-1,3)
    # Aplica a multiplicação de matrizes com a transposta do filtro
    flat_transformed = np.dot(flat_img, t_filter.T)

    # Diminui a intensidade do brilho da imagens
    if bright != 1:
        flat_transformed = np.multiply(flat_transformed, bright)
    
    # Retorna ao formato tridimensional
    transformed_image = flat_transformed.reshape(img.shape[0], img.shape[1], img.shape[2])

    # Limita valores maiores que 255
    transformed_image = np.clip(transformed_image * 255, 0, 255).astype(np.uint8)
    return transformed_image


if __name__ == '__main__':
    old_pele = apply_linear_transform(img=pele, t_filter=old_filter)
    imsave(get_out_path('old_pele'), img_as_ubyte(old_pele))
