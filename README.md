# KEVS

KEVS IPCAI 2025

Setting-up

Just for testing metrics and outcome prediction ability:

1) Clone the KEVS repository https://github.com/tomboucherucl/KEVS.git
2) Download the images from (add url)
3) Run "pip install -e ."
4) Run "python main.py"

For using U-Mamba to make predictions

1. Clone the KEVS repository https://github.com/tomboucherucl/KEVS.git
2. Clone the U-Mamba repository https://github.com/bowang-lab/U-Mamba.git in the KEVS main directory
3. If your CUDA version is not compatible with mamba, please create a Docker container to support this. We provide an example Dockerfile which also contains the necessary paths for the U-Mamba folders.
4. Run Docker file/ run "pip install -e ."
5. Move to the folder U-Mamba/umamba and run "pip install -e ."
6. Move model weights (add url) to the U-Mamba "Results" folder.
7. Uncomment #predict_vat() in main.py
8. Download the scans and abdominal cavity masks from (add url)
9. run main.py
