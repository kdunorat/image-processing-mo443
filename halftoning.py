import numpy as np
from skimage import io
from skimage.util import img_as_ubyte

technics = {
    "floyd_steinberg": [
        (( 1,  0), 7/16), ((-1, 1), 3/16), (( 0, 1), 5/16), (( 1, 1), 1/16)
    ],
    "stevenson_arce": [
        (( 2, 0), 32/200),
        ((-2, 1), 12/200), ((-1, 1), 26/200), (( 0, 1), 30/200),
        (( 1, 1), 26/200), (( 2, 1), 12/200),
        ((-2, 2),  5/200), ((-1, 2), 12/200), (( 0, 2), 12/200),
        (( 1, 2), 12/200), (( 2, 2),  5/200)
    ],
    "burkes": [
        (( 1, 0), 8/32), (( 2, 0), 4/32),
        ((-2, 1), 2/32), ((-1, 1), 4/32), (( 0, 1), 8/32),
        (( 1, 1), 4/32), (( 2, 1), 2/32)
    ],
    "sierra": [
        (( 1, 0), 5/32), (( 2, 0), 3/32),
        ((-2, 1), 2/32), ((-1, 1), 4/32), (( 0, 1), 5/32),
        (( 1, 1), 4/32), (( 2, 1), 2/32),
        ((-1, 2), 2/32), (( 0, 2), 3/32), (( 1, 2), 2/32)
    ],
    "stucki": [
        (( 1, 0), 8/42), (( 2, 0), 4/42),
        ((-2, 1), 2/42), ((-1, 1), 4/42), (( 0, 1), 8/42),
        (( 1, 1), 4/42), (( 2, 1), 2/42),
        ((-2, 2), 1/42), ((-1, 2), 2/42), (( 0, 2), 4/42),
        (( 1, 2), 2/42), (( 2, 2), 1/42)
    ],
    "jarvis_judice_ninke": [
        (( 1, 0), 7/48), (( 2, 0), 5/48),
        ((-2, 1), 3/48), ((-1, 1), 5/48), (( 0, 1), 7/48),
        (( 1, 1), 5/48), (( 2, 1), 3/48),
        ((-2, 2), 1/48), ((-1, 2), 3/48), (( 0, 2), 5/48),
        (( 1, 2), 3/48), (( 2, 2), 1/48)
    ]
}

def error_diffusion(img, technic_name, zigue_zague = False):
    """Aplica difusão de erro (0/255) usando a tecnica escolhida."""
    halftoning_tech = technics[technic_name]
    height, width = img.shape
    new_image = img.astype(np.float32).copy()

    for y in range(height):
        if zigue_zague:
            if y % 2 == 0: # Verifica se a linha é par
                left2right = True
            else:
                left2right = False
        else:
            left2right = True

        # Define se o range é da esquerda pra direita ou da direira pra esquerda
        sweep_range = range(width) if left2right else range(width - 1, -1, -1)

        for x in sweep_range:
            old = new_image[y, x]
            new = 255 if old >= 128 else 0
            new_image[y, x] = new
            error = old - new

            # espalha o erro
            for (x_neighbor, y_neighbor), weight in halftoning_tech:
                new_x = x + (x_neighbor if left2right else -x_neighbor) # se a varredura é invertida, espelha o x do vizinho
                new_y = y + y_neighbor
                if 0 <= new_x < width and 0 <= new_y < height:
                    new_image[new_y, new_x] += error * weight

    # retorna já normalizada
    return new_image / 255.0

def apply_all_halftones(img, zigue_zague = True):
    # Executa as 6 tecnicas e devolve um dicionário {nome_tecnica: imagem_resultante}
    result = {}
    for name in technics:
        result[name] = error_diffusion(img, name, zigue_zague)
    return result


if __name__ == "__main__":
    baboon = io.imread("./images/baboon_monocromatica.png")   # uint8 grayscale
    results = apply_all_halftones(baboon)
    # salvando
    for name, img_ht in results.items():
        io.imsave(f"./output_t2/{name}.png", img_as_ubyte(img_ht))
