import matplotlib.pyplot as plt
import numpy as np

baboon_mono = plt.imread('images/baboon_monocromatica.png')
mosaic_order = np.array([6,11,13,3,8,16,1,9,12,14,2,7,4,15,10,5])


def get_coordinates(index, image_size, grid):
    """Retorna as coordenadas do pedaço de n pixels de indíce especificado
    index: indice do pedaço
    grid: tamanho do pedaço
    image_size: tamanho da imagem original
    """
    chunk_size = image_size// grid
    row = (index - 1) // grid 
    col = (index - 1) % grid

    x_start = col*chunk_size
    x_end = (col+1)*chunk_size
    y_start = row*chunk_size
    y_end = (row+1)*chunk_size

    return [y_start, y_end, x_start, x_end]


if __name__ == '__main__':
    baboon_mosaic = np.zeros(baboon_mono.shape)
    for index, m in enumerate(mosaic_order):
        chunk_coord = get_coordinates(index=m, image_size=baboon_mono.shape[0], grid=4)
        mosaic_coord = get_coordinates(index=index+1, image_size=baboon_mono.shape[0], grid=4)
        baboon_mosaic[mosaic_coord[0]:mosaic_coord[1], mosaic_coord[2]:mosaic_coord[3]] = baboon_mono[chunk_coord[0]:chunk_coord[1], chunk_coord[2]:chunk_coord[3]]

    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.imshow(baboon_mono, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(baboon_mosaic, cmap='gray')
    plt.title('Mosaico')
    plt.axis('off')

    plt.show()
