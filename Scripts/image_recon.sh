folders=(
    "arc_10"
    "arc_30"
    "random_10"
    "random_30"
)

for folder in "${folders[@]}"; do
    python reconstruction_TV.py --input_path ./Experiments/${folder}/Visibilities --output_path ./Experiments/${folder}/TV-Reconstruction
    python reconstruction_MEM.py --input_path ./Experiments/${folder}/Visibilities --output_path ./Experiments/${folder}/MEM-Reconstruction
    python reconstruction_CS.py --input_path ./Experiments/${folder}/Visibilities --output_path ./Experiments/${folder}/CS-Reconstruction
done