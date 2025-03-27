import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from utils import verifica_dtype, gaussian_kernel, bits_por_pixel

# Ler a imagem
watch = plt.imread('images/watch.png')

# Verifica intervalo de pixels
#verifica_dtype(watch)

# Transformar em escala de cinza
gray_watch = 0.2126 * watch[:, :, 0] + 0.7152 * watch[:, :, 1] + 0.0722 * watch[:, :, 2]

# Criando o kernel
kernel = gaussian_kernel(21, 5)

# Cria um padding na imagem refletindo os pixels
padded_watch = np.pad(gray_watch, pad_width= kernel.shape[0] // 2, mode='reflect')

# Altura e largura do kernel
kH, kW = kernel.shape

# Cria uma "janela deslizante" da imagem com todas as regiões onde o kernel será aplicado.
# A função sliding_window_view gera uma matriz de shape (H - kH + 1, W - kW + 1, kH, kW)
windows = np.lib.stride_tricks.sliding_window_view(padded_watch, (kH, kW))

# Realiza a multiplicação elemento a elemento de cada janela com o kernel e soma os resultados
blurred_watch = np.sum(windows * kernel, axis=(-1, -2))


# Aplica a convolve2d do scipy para comparação
blurred_2 = convolve2d(gray_watch, kernel, mode='same')


# Realçe dos contornos
watch_sketch = np.divide(gray_watch, blurred_watch)

# Evita a perda de contraste
watch_sketch = np.clip(watch_sketch, 0, 1)

plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1)
plt.imshow(watch)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(gray_watch, cmap='gray')
plt.title('Escala de cinza')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(blurred_watch, cmap='gray')
plt.title('Imagem com desfoque')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(watch_sketch, cmap='gray')
plt.title('Processamento Final')
plt.axis('off')

plt.tight_layout()
plt.show()