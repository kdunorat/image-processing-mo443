import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from change_color import apply_linear_transform
from utils import verifica_dtype, plot_image


vermeer = io.imread('images/vermeer.png')

old_filter = np.array([[0.393, 0.769, 0.189],[0.349, 0.686, 0.168],[0.272, 0.534, 0.131]])

old_vermeer = apply_linear_transform(vermeer, old_filter)
vermeer_one_channel = 0.2989 * old_vermeer[:, :, 0] + 0.5870 * old_vermeer[:, :, 1] + 0.1140 * old_vermeer[:, :, 2]
v = verifica_dtype(vermeer_one_channel)

# Plota individualmente
plot_image(vermeer, title='Original', color=True)
plot_image(vermeer_one_channel, title='1 canal', color=False)
plot_image(old_vermeer, title='Cores Alteradas', color=False)

# # Plotando lado a lado
# plt.figure(figsize=(16, 4))
# plt.subplot(1,3,1)
# plt.imshow(vermeer, cmap='gray')
# plt.title('Original')
# plt.axis('off')
#
# plt.subplot(1,3,2)
# plt.imshow(vermeer_one_channel, cmap='gray')
# plt.title('1 canal')
# plt.axis('off')
#
# plt.subplot(1,3,3)
# plt.imshow(old_vermeer)
# plt.title('Cores Alteradas')
# plt.axis('off')
# plt.show()
