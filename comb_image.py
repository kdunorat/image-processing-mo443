from skimage import io, color
from skimage.transform import resize
from utils import plot_image, plot_in_grid

crimson = io.imread('images/crimson.png')
division_bell = io.imread('images/division_bell.png')
crimson_mono = color.rgb2gray(crimson)
division_bell_mono = color.rgb2gray(division_bell)
# Ajustando o tamanho de uma das imagens
crimson_mono_resized = resize(crimson_mono, division_bell_mono.shape)


def image_comb(img_a, img_b, comb_tuple: tuple):
    a, b = comb_tuple
    final_image  = a*img_a + b*img_b

    return final_image

crimson_lisa_20_80 = image_comb(crimson_mono_resized, division_bell_mono, (0.2, 0.8))
crimson_lisa_50_50 = image_comb(crimson_mono_resized, division_bell_mono, (0.5, 0.5))
crimson_lisa_80_20 = image_comb(crimson_mono_resized, division_bell_mono, (0.8, 0.2))
images = [
    crimson_mono_resized, division_bell_mono,
    crimson_lisa_20_80, crimson_lisa_50_50, crimson_lisa_80_20
]
titles = [
    'Imagem A (crimson)', 'Imagem B (division_bell)',
    '0.2*A + 0.8*B', '0.5*A + 0.5*B', '0.8*A + 0.2*B'
]
# Exibir em grade com 3 colunas
plot_in_grid(images, titles=titles, n_columns=3, size=(16, 8))
