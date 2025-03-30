import matplotlib.pyplot as plt
import numpy as np
from utils import verifica_dtype, bits_por_pixel

baboon_mono = plt.imread('images/baboon_monocromatica.png')
baboon_mono = np.clip(baboon_mono * 255, 0, 255).astype(np.uint8)


bit_channels_list = []

for i in range(8):  # Para cada bit de 0 a 7
    # Desloca os bits da imagem i posições para a direita
    shifted = baboon_mono >> i

    # Faz um And com 1 para isolar o bit menos significativo por pixel
    bit_channel = shifted & 1

    # Adiciona o resultado (uma imagem binária) à lista
    bit_channels_list.append(bit_channel)

# Mostrar os 8 planos de bits
plt.figure(figsize=(16, 4))
for i, bp in enumerate(bit_channels_list):
    plt.subplot(1, 8, i + 1)
    plt.imshow(bp, cmap='gray')
    plt.title(f'Plano de bit {i}')
    plt.axis('off')
plt.tight_layout()
plt.show()
