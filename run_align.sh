#!/usr/bin/env bash
set -e                                  # aborta se algum comando falhar

INPUT_DIR="imagens_inclinadas"
OUTPUT_DIR="output_t3"

mkdir -p "$OUTPUT_DIR"                  # garante que a pasta existe

# lista das 7 imagens (exatamente como aparecem no print)
images=(
  "neg_4.png"
  "neg_28.png"
  "partitura.png"
  "pos_24.png"
  "pos_41.png"
  "sample1.png"
  "sample2.png"
)

for img in "${images[@]}"; do
  base="${img%.*}"                      # remove extensão

  # alinhamento por perfil horizontal
  python alinhar.py "$INPUT_DIR/$img" horizontal \
                    "$OUTPUT_DIR/${base}_horizontal.png"

  # alinhamento por Hough
  python alinhar.py "$INPUT_DIR/$img" hough \
                    "$OUTPUT_DIR/${base}_hough.png"
done
