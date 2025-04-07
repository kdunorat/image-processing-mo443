from skimage import io, img_as_ubyte
from skimage.io import imsave
from utils import plot_in_grid, get_out_path

baboon_mono = io.imread('images/baboon_monocromatica.png')


# Função de quantização
def quantize(img, levels):
    fator = 256 // levels
    return (img // fator) * fator

# Lista dos niveis
levels_list = [256, 128, 64, 32, 16, 8, 4, 2]
images = [baboon_mono if n == 256 else quantize(baboon_mono, n) for n in levels_list]

titles = [f'{l} níveis' for l in levels_list]

# Plota as imagens numa grade 3x3
plot_in_grid(images, titles=titles, n_columns=3)

for img, title in zip(images, titles):
    imsave(get_out_path(title), img_as_ubyte(img))