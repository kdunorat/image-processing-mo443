from skimage import io, img_as_ubyte
from skimage.io import imsave
import numpy as np
from utils import verifica_dtype, get_out_path


city = io.imread('images/city.png')
print(verifica_dtype(city))
# Negativo
negative_city = 255 - city

# Redução de intensidade
city_100_200 = np.clip(city, 100, 200)

# Inversão das linhas pares
linha_0 = city[0, :]
linha_0_reverse = linha_0[::-1]

# Atribui o reverso das linhas pares para as linhas pares
city_pair_reverse = city.copy()
city_pair_reverse[::2, :] = city_pair_reverse[::2, ::-1]

# Espelhamento horizontal
city_reflex_full = city.copy()
city_half_size = city.shape[0] // 2
city_reflex_half = city[:city_half_size, :]
city_reflex_half = city_reflex_half[::-1, :]
city_reflex_full[city_half_size:, :] = city_reflex_half

# Espelhamento vertical
city_upside = city[::-1, :]

# Lista de imagens e títulos
images = [
    city, negative_city, city_100_200,
    city_pair_reverse, city_reflex_full, city_upside
]

titles = [
    'Original', 'Negativo', 'Intensidade 100–200',
    'Linhas pares invertidas', 'Espelhamento horizontal', 'Espelhamento vertical'
]

# Salvando
for img, title in zip(images, titles):
    imsave(get_out_path(title), img_as_ubyte(img))
