from skimage import io, img_as_ubyte
from skimage.io import imsave
import numpy as np
from change_color import apply_linear_transform
from utils import get_out_path


vermeer = io.imread('../images/vermeer.png')

old_filter = np.array([[0.393, 0.769, 0.189],[0.349, 0.686, 0.168],[0.272, 0.534, 0.131]])

old_vermeer = apply_linear_transform(vermeer, old_filter)
vermeer_one_channel = 0.2989 * old_vermeer[:, :, 0] + 0.5870 * old_vermeer[:, :, 1] + 0.1140 * old_vermeer[:, :, 2]
# Normaliza para salvar como PNG
vermeer_one_channel = vermeer_one_channel / 255.0


# Plotando lado a lado
imsave(get_out_path('old_vermeer'), img_as_ubyte(old_vermeer))
imsave(get_out_path('vermeer_one_channel'), img_as_ubyte(vermeer_one_channel))
