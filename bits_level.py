from skimage import io
from utils import plot_in_grid

baboon_mono = io.imread('images/baboon_monocromatica.png')


bit_channels_list = []
titles = []

for i in range(8):
    # Desloca os bits
    shifted = baboon_mono >> i
    # Isola o bit menos significativo
    bit_channel = shifted & 1
    bit_channels_list.append(bit_channel)
    # Titulos das imagens
    titles.append(f'Plano de bits {i}')

# Exibição em grade
plot_in_grid(bit_channels_list, titles=titles, n_columns=4, size=(10, 6))
