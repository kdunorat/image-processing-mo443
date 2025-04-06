import matplotlib.pyplot as plt
from skimage import io, color


crimson = io.imread('images/crimson.png')
monalisa = io.imread('images/monalisa.png')
crimson_mono = color.rgb2gray(crimson)
monalisa_mono = color.rgb2gray(monalisa)

def image_comb(img_a, img_b, comb_tuple: tuple):
    a, b = comb_tuple
    final_image  = a*img_a + b*img_b

    return final_image

crimson_mono = image_comb(crimson_mono, monalisa_mono, (0.5, 0.5))
plt.imshow(crimson_mono, cmap='gray')
plt.axis('off')
plt.show()