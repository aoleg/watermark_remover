# watermark_remover
Multi-GPU enabled fast batch removal of watermarks from large image datasets

Key Architectural Features:
- Multi GPU (AI) and CPU (I/O) workloads for high throughput.
- Pause/resume processing at any time
- Per-GPU performance/status tracking in a pretty console display

On a dual RTX 4090 machine, you can de-watermark over 1000 images per minute. ⚡😎⚡

<img width="1077" height="447" alt="image" src="https://github.com/user-attachments/assets/b8105659-16b2-4f04-b367-a6979c4a5a46" />

Enjoy!

# Install instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/jferments/watermark_remover.git
    ```

2. Enter the project directory
    ```bash
    cd watermark_remover
    ```

3. Install required Python packages
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install rich ultralytics simple-lama-inpainting opencv-python torch transformers --upgrade
    ```
    `transformers` is only needed for DINOv3-YOLOv12 detector checkpoints (see below). On Windows, install a CUDA build of torch from https://pytorch.org/get-started/locally/ instead of the default PyPI wheel, which is CPU-only.
4. Download [fancyfeast](https://huggingface.co/fancyfeast)'s custom [YOLOv11 watermark detection model](https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection) checkpoint from Hugging Face:
    ```bash
    wget https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection/resolve/main/yolo11x-train28-best.pt
    ```

5. Optional: download [corzent](https://huggingface.co/corzent)'s [YOLOv12x + DINOv3 watermark detection model](https://huggingface.co/corzent/yolov12x-dino3-watermark-detection). It was trained with the [DINOV3-YOLOV12](https://github.com/Sompote/DINOV3-YOLOV12) fork of ultralytics, and this project ships a small compatibility layer (`dino3_compat.py`) so it runs on stock ultralytics. It needs `transformers` and roughly twice the inference time of the YOLOv11 model.
    ```bash
    wget https://huggingface.co/corzent/yolov12x-dino3-watermark-detection/resolve/main/yolov12x-dino3-watermark-detection.pt
    ```

# Usage instructions

**Basic usage**

    python3 watermark_remover.py -i /path/to/inputs -o /path/to/outputs -R

**Processing each first-level subfolder into its own output folder (Windows)**

    recursive_scan.bat D:\datasets --png -R

This runs the script once per subfolder of `D:\datasets`, writing `D:\datasets\NAME` to `D:\datasets\_NAME`. All parameters after the folder are passed through unchanged. Subfolders whose name starts with `_` are skipped, so the command can be re-run and resumes per folder.

**Using the DINOv3-YOLOv12 detector**

    python3 watermark_remover.py -i /path/to/inputs -o /path/to/outputs -R -w yolov12x-dino3-watermark-detection.pt



## Pausing and Resuming

For very large datasets, you can safely stop the script at any time by pressing **`Ctrl+C`**. The script will perform a graceful shutdown, save its progress, and print a summary for the session.

When you run the script again with the **exact same output directory**, it will automatically detect the `.processing_log.txt` checkpoint file and resume where it left off, skipping any images that were already successfully processed.

<img width="688" height="219" alt="image" src="https://github.com/user-attachments/assets/b17a43a2-23e1-40a3-9ae9-91a8f1e9236d" />

## Command-Line Arguments

Here is a detailed explanation of all available arguments:

* **`-i, --input`** `<path>` **(Required)** Specifies the path to the folder containing the images you want to process.

* **`-o, --output`** `<path>` **(Required)** Specifies the path to the folder where the clean, processed images will be saved. The original directory structure from the input will be replicated here. This directory also stores the `.processing_log.txt` file for resuming sessions.

* **`-w, --weights`** `<path>`  
    Specifies the path to the YOLOv11 model weights file.  
    (Default: `yolo11x-train28-best.pt`)

* **`--conf`** `<float>`  
    The confidence threshold for the YOLO object detection model (from `0.0` to `1.0`). Lower values will detect more potential watermarks but may also have more false positives.  
    (Default: `0.1`)

* **`--dilate`** `<integer>`  
    The number of pixels to expand (dilate) the detected watermark mask. This is useful for ensuring the inpainting model covers any faint "glow" or aliasing around the edges of a watermark. Set to `0` to disable.  
    (Default: `15`)

* **`--inpaint-max-size`** `<integer>`  
    LaMa does not inpaint the full image. It inpaints each watermark with a margin of context around it, and only the masked pixels are replaced. This option sets the longest side, in pixels, of such an area. A larger area is scaled down for inpainting, and the result is scaled back up. LaMa needs about 0.9 GB of VRAM per megapixel, so `0` (no limit) can overfill VRAM on large images.  
    (Default: `2048`)

* **`-R, --recursive`** A flag that, if present, tells the script to search for images in all subdirectories of the input folder. If omitted, it will only process images in the top-level directory.

* **`--cpu-workers`** `<integer>`  
    The total number of CPU processes to spawn for the I/O-bound task of writing image files to disk. By default, it uses all available CPU cores to maximize I/O throughput.  
    (Default: Your system's CPU core count)

* **`--debug`** A flag that, if present, will save two intermediate images for each detected watermark into an `output/debug/` directory, named after the full input file name (for example `photo.jpg_mask_raw.png`):
    1.  `_mask_raw.png`: The raw black and white mask.
    2.  `_mask_preview.png`: The mask overlaid in semi-transparent red on the original image.

* **`--png`** A flag that, if present, saves every output image as lossless PNG, regardless of the input format. `photo.jpg` is written as `photo.png`. When several images in one folder would get the same PNG name (for example `photo.jpg`, `photo.png` and `photo.webp`), a PNG source keeps its name and the others keep their extension: `photo.jpg.png`, `photo.webp.png`. Names are compared without case, as on Windows.

* **`--skip-clean`** `on|off`  
    Controls what happens to images in which no watermark is detected. `on`, `yes` or `1` does not write them to the output folder. `off`, `no` or `0` writes them too, so the output folder is a complete copy of the input. JPEG and WebP files are then copied byte for byte, so they lose no quality. Lossless sources (PNG, BMP, TIFF) are re-encoded, and with `--png` every file is re-encoded to PNG. Images with a watermark are always written. Skipped images are still recorded in `.processing_log.txt`, so a resumed session does not scan them again. To write them later, delete the log file or use a different output folder. `--skip-clean` with no value means `on`.  
    (Default: `on`)


## Notes

* There will be some amount of false negatives (watermarks that don't get detected/removed) and some false positives (image features incorrectly identified as watermarks and removed). My intitial tests were on an image dataset that was heavily weighted towards a specific type of watermark that almost always appears in the corners of the images. With these types of images @ the default settings, the false negative rate was very low (less than 1-2%). You can play around with the *--conf* command line option to adjust how sensitive watermark detection is (a lower # will decrease false negatives at the cost of increased false positives, and vice versa). Or if you have an image dataset with a significantly different type of watermarks, you can use a different watermark detection model that is better at detecting those kinds of marks and use it with this script. 
* I have only tested this on my own machine which has dual 4090s and an AMD 7965WX (24 core) CPU, and I was averaging about 1000-1200 images/minute. Depending on your hardware (especially if you are running CPU only) or the size/resolution of images you're working with, you might experience much lower speeds.
* I have written more about the rationale for creating this script [here](https://jferments.medium.com/large-scale-batch-removal-of-watermarks-from-image-datasets-d7fb5ab226b0).

## Disclaimer
 This script is provided for educational and technical demonstration purposes only. Removing watermarks from images may violate copyright or intellectual property rights. Users of this script are solely responsible for ensuring they have the legal right to modify the images they process. The author assumes no liability for misuse of this tool.
