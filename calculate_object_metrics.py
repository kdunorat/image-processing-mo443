from skimage import io, img_as_ubyte, color, measure
import matplotlib.pyplot as plt


def analisar_objetos_skimage(img, name):
    """
    Função principal que realiza a análise de objetos em uma imagem,
    utilizando a biblioteca Scikit-image.
    """

    gray_img = color.rgb2gray(img)
    # Binarizando
    threshold = 0.8
    img_bin = gray_img < threshold
    labeled_img = measure.label(img_bin)

    regions = measure.regionprops(labeled_img)
    contours = measure.find_contours(labeled_img, 0.5)

    # --- Extração e Impressão de Propriedades ---
    areas = []
    print(f"Resultados da imagem {name}")
    print(f"número de regiões: {len(regions)}\n")

    for i, region in enumerate(regions):
        # Extrair propriedades diretamente do objeto 'region'
        area = region.area
        areas.append(area)
        perimeter = region.perimeter
        excentricity = region.eccentricity
        solidity = region.solidity

        print(f"região {i}:")
        print(f"  área: {area}")
        print(f"  perímetro: {perimeter:.6f}")
        print(f"  excentricidade: {excentricity:.6f}")
        print(f"  solidez: {solidity:.6f}")

    print("\n--- Classificação por Área ---")


    pequenos = sum(1 for a in areas if a < 1500)  #
    medios = sum(1 for a in areas if 1500 <= a < 3000)  #
    grandes = sum(1 for a in areas if a >= 3000)  #

    print(f"número de regiões pequenas: {pequenos}")
    print(f"número de regiões médias: {medios}")
    print(f"número de regiões grandes: {grandes}")

    # Salvando os contornos
    cont_fig, ax = plt.subplots(figsize=(6, 6))
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], linewidth=2, color='red')
    ax.set_aspect('equal')
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(0, img.shape[0])
    ax.invert_yaxis()
    ax.axis('off')
    cont_fig.savefig(f'./output_t4/contornos-{name}.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(cont_fig)

    # Salvando as imagens rotuladas
    labels_fig, ax_labels = plt.subplots()
    ax_labels.imshow(img)
    for r in regions:
        y, x = r.centroid
        ax_labels.text(x, y, str(r.label - 1), color='black', fontsize=12, ha='center', va='center')
    ax_labels.axis('off')
    labels_fig.savefig(f'./output_t4/regioes_rotuladas-{name}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(labels_fig)

    # Salvando o histograma de areas
    max_area = max(areas) if areas else 0
    # O limite superior será o maior valor entre a (maior área + 500) e 3500, garantindo que seja sempre > 3000
    upper_bound = max(max_area + 500, 3500)
    bins = [0, 1500, 3000, upper_bound]
    hist_fig, ax_hist = plt.subplots()
    ax_hist.hist(areas, bins=bins, color='blue', edgecolor='black')
    ax_hist.set_title('Histograma de Áreas')
    ax_hist.set_xlabel('Área')
    ax_hist.set_ylabel('Número de Objetos')
    ax_hist.grid(axis='y', alpha=0.75)
    hist_fig.savefig(f'./output_t4/histograma_areas-{name}.png', dpi=300, bbox_inches='tight')
    plt.close(hist_fig)


if __name__ == '__main__':
    for i in range(3):
        objects_img = io.imread(f'./images/objetos{i+1}.png')
        analisar_objetos_skimage(objects_img, name=f'objetos{i}')
