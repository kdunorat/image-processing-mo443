import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_ubyte


def gerar_espectro_fourier(imagem, plotar=True):
    """
    Gera o espectro de Fourier de uma imagem 2D.

    Parâmetros:
    - imagem: matriz 2D numpy array (imagem em escala de cinza)
    - plotar: se True, exibe o espectro de magnitude

    Retorna:
    - magnitude_log: espectro de magnitude em escala logarítmica
    - fase: matriz de fase
    """

    # 1. Aplicar a FFT 2D
    fft2 = np.fft.fft2(imagem)

    # 2. Centralizar o zero (frequência baixa no centro)
    fft2_shift = np.fft.fftshift(fft2)

    # 3. Calcular magnitude e fase
    magnitude = np.abs(fft2_shift)
    fase = np.angle(fft2_shift)

    # 4. Aplicar log para melhorar a visualização da magnitude
    # magnitude_log = np.log(1 + magnitude)
    magnitude_log = magnitude

    if plotar:
        plt.figure(figsize=(10, 4))

        # Espectro de Magnitude
        plt.subplot(1, 2, 1)
        plt.imshow(magnitude_log, cmap='gray')
        plt.title('Espectro de Magnitude (log)')
        plt.axis('off')

        # Espectro de Fase
        plt.subplot(1, 2, 2)
        plt.imshow(fase, cmap='gray')
        plt.title('Espectro de Fase')
        plt.axis('off')

        plt.show()

    return magnitude_log, fase, fft2_shift

if __name__ == '__main__':
    baboon = io.imread('./images/baboon_monocromatica.png')
    test_image = np.full((512,512), 0.5, dtype=np.float32)
    print(test_image)
    # plt.figure(figsize=(10, 4))
    # plt.imshow(test_image, cmap='gray', vmin=0, vmax=1)
    # plt.title('Teste')
    # plt.axis('off')
    # plt.show()
    m, fase, fft = gerar_espectro_fourier(test_image)
    print(fft)