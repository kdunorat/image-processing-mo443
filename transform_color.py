import matplotlib.pyplot as plt
import numpy as np
from altera_cores import apply_linear_transform
from utils import verifica_dtype

pele = plt.imread('images/1970_copa_pele_abraco.jpg')

old_filter = np.array([[0.393, 0.769, 0.189],[0.349, 0.686, 0.168],[0.272, 0.534, 0.131]])

old_pele = apply_linear_transform(pele, old_filter)
old_pele_one_channel = 0.2989 * old_pele[:, :, 0] + 0.5870 * old_pele[:, :, 1] + 0.1140 * old_pele[:, :, 2]
v = verifica_dtype(old_pele_one_channel)

plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.imshow(old_pele_one_channel, cmap='gray')
plt.axis('off')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(old_pele)
plt.axis('off')
plt.show()
