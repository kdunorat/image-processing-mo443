import matplotlib.pyplot as plt
import numpy as np


city = plt.imread('images/city.png')
city = np.clip(city * 255, 0, 255)

# Negativo
negative_city = abs(city-255)

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

plt.imshow(city_upside, cmap='gray')
plt.axis('off')
plt.show()
