import matplotlib.pyplot as plt
import numpy as np


baboon_mono = plt.imread('images/baboon_monocromatica.png')
butterfly = plt.imread('images/butterfly.png')

def image_comb(img_a, img_b, comb_tuple: tuple):
    a, b = comb_tuple
    final_image  = a*img_a + b*img_b

    return final_image


babooonfly = image_comb(baboon_mono, butterfly, (0.5, 0.5))
plt.imshow(babooonfly, cmap='gray')
plt.axis('off')
plt.show()