from skimage import io, img_as_ubyte
from skimage.io import imsave
from utils import plot_in_grid, get_out_path

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

# Salva 1 por 1:
for img, title  in zip(bit_channels_list, titles):
    # Escalar as imagens para plotar
    img = img * 255
    imsave(get_out_path(title), img)
