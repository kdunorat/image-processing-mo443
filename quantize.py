import matplotlib.pyplot as plt
import numpy as np

baboon_mono = plt.imread('images/baboon_monocromatica.png')
baboon_mono = np.clip(baboon_mono * 255, a_min=0, a_max=255).astype(np.uint8)

def quantize_img(img, levels: int):
    pass
