import matplotlib.pyplot as plt
import numpy as np

# Ler a imagem
baboon_mono = plt.imread('images/baboon_monocromatica.png')

def gamma_correction(image, gamma):
    # Normaliza para [0, 1]
    image_norm = np.clip(image, 0, 1)

    # Aplica a fórmula à imagem e retorna o resultado
    corrected = np.power(image_norm, 1/gamma)
    return corrected

# Testar diferentes gamas
gammas = [0.5, 1.5, 2.5, 3.5]
results = [gamma_correction(baboon_mono, g) for g in gammas]


plt.figure(figsize=(16, 4))
plt.subplot(1, 5, 1)
plt.imshow(baboon_mono, cmap='gray')
plt.title('Original')
plt.axis('off')

# Plota as imagens lado a lado
for i, (g, img_corr) in enumerate(zip(gammas, results)):
    plt.subplot(1, 5, i+2)
    plt.imshow(img_corr, cmap='gray')
    plt.title(f'Gamma = {g}')
    plt.axis('off')

plt.tight_layout()
plt.show()



