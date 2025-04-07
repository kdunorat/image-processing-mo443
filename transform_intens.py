from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_in_grid, plot_image, verifica_dtype

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
plot_image(negative_city)
# Plota em grade com 3 colunas
plot_in_grid(images, titles=titles, n_columns=3, size=(15, 10))
