import glob
import os
import easyocr


def run_ocr(all_images):
    # Cria reader, roda uma vez
    reader = easyocr.Reader(['en'])
    results_dict = dict()
    for image_path in all_images:
        if 'partitura' in image_path:
            continue
        image_name = image_path.split('/')[-1]
        result = reader.readtext(image_path)
        complete_text = " ".join(text for _, text, _ in result)

        results_dict[image_name] = complete_text

    return results_dict


if __name__ == '__main__':
    skewed_path = './imagens_inclinadas'
    align_path = './output_t3'
    all_images = glob.glob(os.path.join(skewed_path, '*'))
    original_names = [os.path.basename(x).split('.')[0] for x in all_images if 'partitura' not in x]
    all_images.extend(glob.glob(os.path.join(align_path, '*')))
    results_dict = run_ocr(all_images)

    for base in original_names:
        desalinhada = results_dict.get(f"{base}.png")
        horizontal = results_dict.get(f"{base}_horizontal.png")
        hough = results_dict.get(f"{base}_hough.png")

    # Para ver o texto completo:
        print(f"{base}:")
        print(f"  \nDesalinhada: {desalinhada}\n")
        print(f"  Horizontal:  {horizontal}\n")
        print(f"  Hough:       {hough}\n")



