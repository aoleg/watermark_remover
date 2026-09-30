#!/usr/bin/env python3
"""
watermark_remover.py — DETECT (fancyfeast YOLOv11 finetune) + ERASE (Simple-LaMa)
======================================================================

Multi-GPU enabled fast batch removal of watermarks from large image datasets

Key Architectural Features:
- Multi GPU (AI) and CPU (I/O) workloads for true parallelism and high throughput.
- Resumable processing via a `.processing_log.txt` checkpoint file.
- Per-GPU status tracking in a pretty Rich console display

--------------------------------------------------------------------
USAGE INSTRUCTIONS
--------------------------------------------------------------------

   python3 watermark_remover.py -i /path/to/inputs -o /path/to/outputs -R

--------------------------------------------------------------------

 ╭────────────────────────────────────────────────────────────────────────╮
 │                  DISCLAIMER & RESPONSIBLE USE NOTICE                   │
 ╰────────────────────────────────────────────────────────────────────────╯
 This script is provided for educational and technical demonstration
 purposes only. Removing watermarks from images may violate copyright
 or intellectual property rights. Users of this script are solely
 responsible for ensuring they have the legal right to modify the images
 they process. The author assumes no liability for misuse of this tool.

"""

# ─────────────────────────────────── Standard Library Imports ───────────────────────────────────
import os
import sys
import argparse
import pathlib
import signal
import time
from collections import deque
import multiprocessing as mp
import subprocess
import queue # For queue.Empty exception

# ────────────────────────────────── Third-Party Library Imports ─────────────────────────────────
try:
    from rich.console import Console, Group
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    print("Error: The 'rich' library is not found. Please install it with 'pip install rich'")
    sys.exit(1)

# ───────────────────────────────────────────────────────────────────────────────────────────────

IMG_EXTS = ("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff")
# Clean images in these formats are copied byte for byte instead of re-encoded (unless --png), so they lose no quality.
# WebP can also be lossless, but a byte copy is exact either way.
LOSSY_EXTS = ("jpg", "jpeg", "webp")

# ╭───────────────────────── CLI Argument Parsing ─────────────────────────╮
def on_off(value: str) -> bool:
    """argparse type for switches that take on/yes/1 or off/no/0."""
    if value.strip().lower() in ("on", "yes", "1"): return True
    if value.strip().lower() in ("off", "no", "0"): return False
    raise argparse.ArgumentTypeError(f"expected on, yes, 1, off, no or 0, got {value!r}")

def parse_cli_args():
    """Sets up and parses command-line arguments using Python's argparse."""
    parser = argparse.ArgumentParser(
        description="Detect watermarks with a custom YOLOv11 model and in-paint with Simple-LaMa.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, type=pathlib.Path, help="Path to the folder containing watermarked images.")
    parser.add_argument("-o", "--output", required=True, type=pathlib.Path, help="Path to the folder where clean images will be saved.")
    parser.add_argument("-w", "--weights", type=pathlib.Path, default=pathlib.Path("yolo11x-train28-best.pt"), help="Path to the YOLO detector weights (.pt). Supports plain ultralytics checkpoints such as\nyolo11x-train28-best.pt and DINOv3-YOLOv12 checkpoints such as yolov12x-dino3-watermark-detection.pt.")
    parser.add_argument("--conf", type=float, default=0.1, help="YOLO detection confidence threshold.")
    parser.add_argument("--dilate", type=int, default=15, help="Pixel amount to expand detected masks.")
    parser.add_argument("--inpaint-max-size", type=int, default=2048, help="Longest side, in pixels, of an area that LaMa inpaints in one pass. LaMa works on the watermark\nplus a margin of context, not on the full image. A larger area is scaled down for inpainting, and\nonly the masked pixels are scaled back up. LaMa needs about 0.9 GB of VRAM per megapixel, so\n0 (no limit) can overfill VRAM on large images. (Default: 2048)")
    parser.add_argument("--trim", action="store_true", help="Crop the watermark off instead of inpainting it. The image is cut to the largest rectangle that\ncontains no detected watermark (with the --dilate margin). JPEG sources are cropped losslessly\nunless --png is set; the left and top edges then move inwards by up to 31 px to the next JPEG block.")
    parser.add_argument("--trim-min-keep", type=float, default=0.75, metavar="FRACTION", help="With --trim: if the rectangle keeps less than this fraction of the image area, the watermark is\nnot near an edge, and the image is inpainted instead. 0 always trims. (Default: 0.75)")
    parser.add_argument("-R", "--recursive", action="store_true", help="Process images in subdirectories recursively.")
    parser.add_argument("--cpu-workers", type=int, default=os.cpu_count(), help="Total number of CPU processes for writing images to disk.")
    parser.add_argument("--debug", action="store_true", help="Save intermediate mask_raw and mask_preview images for debugging.")
    parser.add_argument("--png", action="store_true", help="Save output images as lossless PNG regardless of the input format.")
    parser.add_argument("--skip-clean", type=on_off, nargs="?", const=True, default=True, metavar="{on,off}", help="on, yes, 1 (default): do not write images in which no watermark is detected. They are\nstill recorded in the checkpoint log, so a resumed session does not scan them again.\noff, no, 0: write them to the output folder too. JPEG and WebP files are copied unchanged;\nlossless sources, and every file when --png is set, are re-encoded.")
    return parser.parse_args()
# ╰─────────────────────────────────────────────────────────────────────────╯

# ╭─────────────────── System, Checkpoint & File Discovery Functions ───────────────────╮
def get_gpu_ids():
    """Detects available NVIDIA GPU IDs by shelling out to `nvidia-smi`."""
    try:
        smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], encoding="utf-8").strip()
        return [int(line) for line in smi_output.splitlines()]
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return []

def load_processed_files(log_file_path: pathlib.Path) -> set:
    """Loads the set of already processed file paths from the checkpoint log."""
    if not log_file_path.exists(): return set()
    with open(log_file_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f)

def find_image_files(input_dir: pathlib.Path, recursive: bool, console: Console):
    """Scans for image files, displaying a live counter of the scan."""
    console.print(f"[*] Scanning for images in: [cyan]{input_dir}[/cyan]")
    image_files = []
    with console.status("[bold green]Searching for images...") as status:
        if recursive:
            dirs_scanned = 0
            for dirpath, _, filenames in os.walk(input_dir):
                dirs_scanned += 1
                for filename in filenames:
                    if filename.lower().endswith(IMG_EXTS):
                        image_files.append(pathlib.Path(dirpath) / filename)
                status.update(f"[bold green]Scanning... Dirs: {dirs_scanned:,} | Images found: {len(image_files):,}")
        else:
            all_items = list(input_dir.iterdir())
            for i, item in enumerate(all_items):
                if item.is_file() and item.suffix.lower().lstrip('.') in IMG_EXTS:
                    image_files.append(item)
                status.update(f"[bold green]Scanning... Items: {i+1}/{len(all_items)} | Images found: {len(image_files):,}")
    return image_files

def png_output_renames(relative_paths: list) -> dict:
    """With --png, a.jpg is written as a.png. When several sources would share one PNG name (a.jpg, a.png, a.webp),
    a PNG source keeps it and the others keep their extension: a.jpg.png, a.webp.png. Names are compared without
    case, as on Windows. Returns {input-relative posix path: output-relative posix path} for renamed sources only.
    It is computed from the whole dataset, not only the unprocessed images, so a resumed session picks the same names."""
    groups = {}
    for rel in relative_paths: groups.setdefault(rel.with_suffix(".png").as_posix().lower(), []).append(rel)
    outputs = {}
    for group in groups.values():
        for rel in group:
            keeps_name = len(group) == 1 or rel.suffix.lower() == ".png"
            outputs[rel] = rel.with_suffix(".png") if keeps_name else rel.with_name(rel.name + ".png")
    # A name can still repeat in a contrived dataset (a.jpg next to a real a.jpg.png, or a.PNG next to a.png); number those.
    # Sources that keep their plain name claim it first, so a real a.jpg.png is not displaced by a renamed a.jpg.
    taken, renames = set(), {}
    for rel in sorted(outputs, key=lambda r: (outputs[r] != r.with_suffix(".png"), r.as_posix())):
        out, n = outputs[rel], 2
        while out.as_posix().lower() in taken: out = outputs[rel].with_name(f"{outputs[rel].stem}_{n}.png"); n += 1
        taken.add(out.as_posix().lower())
        if out != rel.with_suffix(".png"): renames[rel.as_posix()] = out.as_posix()
    return renames

# cv2.imread and cv2.imwrite pass the path to the C runtime as narrow bytes, so on Windows they fail on any name
# outside the ANSI code page (and even on Cyrillic under code page 1251). Python opens the file; OpenCV only decodes.
def read_image(path: pathlib.Path):
    import cv2, numpy as np
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None

def write_image(path: pathlib.Path, img) -> bool:
    import cv2
    ok, encoded = cv2.imencode(path.suffix, img)
    if ok: encoded.tofile(str(path))
    return ok

def largest_clear_rect(boxes, width: int, height: int, max_boxes: int = 20):
    """Returns the largest (x1, y1, x2, y2) rectangle of the image that overlaps no box, or None.
    Each side of the best rectangle lies on the image border or on a box edge, so only those candidates are tested."""
    import numpy as np
    boxes = np.asarray(boxes, dtype=np.int64).reshape(-1, 4)
    if len(boxes) > max_boxes: return None  # a tiled watermark; no useful rectangle is left anyway
    lefts, rights = np.r_[0, boxes[:, 2]], np.r_[width, boxes[:, 0]]
    tops, bottoms = np.r_[0, boxes[:, 3]], np.r_[height, boxes[:, 1]]
    L, R, T, B = np.meshgrid(lefts, rights, tops, bottoms, indexing="ij")
    L, R, T, B = L.ravel(), R.ravel(), T.ravel(), B.ravel()
    area = np.clip(R - L, 0, None) * np.clip(B - T, 0, None)
    hits = ((boxes[:, 0][None] < R[:, None]) & (L[:, None] < boxes[:, 2][None]) &
            (boxes[:, 1][None] < B[:, None]) & (T[:, None] < boxes[:, 3][None])).any(axis=1)
    area[hits] = 0
    best = int(area.argmax())
    return (int(L[best]), int(T[best]), int(R[best]), int(B[best])) if area[best] > 0 else None

def crop_jpeg_losslessly(path: pathlib.Path, rect):
    """Crops a JPEG in the DCT domain, so the kept pixels are not recompressed, and keeps EXIF, ICC and other markers.
    rect is (x1, y1, x2, y2) as OpenCV shows the image, that is after EXIF rotation, while the crop applies to the
    stored image; it is mapped back first. The stored left and top edges must lie on an MCU boundary (8 to 32 px),
    so they move inwards to the next boundary. Returns the new file's bytes, or None if a lossless crop is not possible."""
    import io, turbojpeg
    from PIL import Image
    data = path.read_bytes()
    header = turbojpeg.decompress_header(data)
    w, h = header["width"], header["height"]
    try: orientation = Image.open(io.BytesIO(data)).getexif().get(0x0112, 1)
    except Exception: orientation = 1
    x1, y1, x2, y2 = rect
    stored = {  # displayed edges -> stored edges, per EXIF orientation
        1: (x1, y1, x2, y2),         2: (w - x2, y1, w - x1, y2),         3: (w - x2, h - y2, w - x1, h - y1),
        4: (x1, h - y2, x2, h - y1), 5: (y1, x1, y2, x2),                 6: (y1, h - x2, y2, h - x1),
        7: (w - y2, h - x2, w - y1, h - x1),                              8: (w - y2, x1, w - y1, x2),
    }.get(orientation)
    mcu = {turbojpeg.SAMP.Y444: (8, 8), turbojpeg.SAMP.GRAY: (8, 8), turbojpeg.SAMP.Y422: (16, 8), turbojpeg.SAMP.Y420: (16, 16),
           turbojpeg.SAMP.Y440: (8, 16), turbojpeg.SAMP.Y411: (32, 8), turbojpeg.SAMP.Y441: (8, 32)}.get(header["subsamp"])
    if stored is None or mcu is None: return None
    sx1, sy1, sx2, sy2 = stored
    sx1, sy1 = -(-sx1 // mcu[0]) * mcu[0], -(-sy1 // mcu[1]) * mcu[1]
    if sx2 <= sx1 or sy2 <= sy1: return None
    if (sx1, sy1, sx2, sy2) == (0, 0, w, h): return data
    try: return turbojpeg.transform(data, x=sx1, y=sy1, w=sx2 - sx1, h=sy2 - sy1, crop=True, perfect=False)
    except Exception: return None
# ╰─────────────────────────────────────────────────────────────────────────╯

# ╭─────────────────────── Custom Rich Progress Columns ───────────────────────╮
class SpeedColumn(TextColumn):
    """A custom `rich` progress column for displaying speed in images/sec."""
    def __init__(self, *args, **kwargs): super().__init__(" ", *args, **kwargs)
    def render(self, task) -> Text:
        if task.speed is None: return Text("N/A", style="dim green")
        return Text(f"{task.speed:.2f} img/s", style="green")

class EstimatedTimeRemainingColumn(TimeRemainingColumn):
    """A `TimeRemainingColumn` that adds a custom label."""
    def render(self, task) -> Text:
        if task.finished or task.time_remaining is None: return Text("-:--:--", style="dim")
        return Text("Est. time remaining: ", style="dim") + super().render(task)
# ╰─────────────────────────────────────────────────────────────────────────╯

# ╭─────────────────────── GPU & CPU Worker Implementations ───────────────────────╮
def inpaint_areas(lama_model, img_bgr, mask, max_size: int):
    """Inpaints each masked area in a crop with context around it, instead of the full image. LaMa's VRAM use grows
    with the pixel count, so a 40 MP image does not fit in 32 GB. A crop whose longest side exceeds max_size is scaled
    down for LaMa and its result scaled back up. Only masked pixels are replaced; every other pixel stays unchanged."""
    import cv2, numpy as np
    from PIL import Image
    H, W = mask.shape
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rects = []
    for x, y, w, h, _ in stats[1:]:
        margin = max(128, max(w, h) // 2)  # LaMa fills the hole from its surroundings, so give it some
        rects.append([max(0, x - margin), max(0, y - margin), min(W, x + w + margin), min(H, y + h + margin)])
    merged = True  # overlapping crops are merged, so no crop sees an unfilled part of another watermark as context
    while merged:
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    rects[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]; del rects[j]; merged = True; break
            if merged: break
    result = img_bgr.copy()
    for x1, y1, x2, y2 in rects:
        crop, crop_mask = img_bgr[y1:y2, x1:x2], mask[y1:y2, x1:x2]
        h, w = crop_mask.shape
        scale = min(1.0, max_size / max(h, w)) if max_size > 0 else 1.0
        if scale < 1.0:
            size = (max(8, round(w * scale)), max(8, round(h * scale)))
            crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
            lama_mask = (cv2.resize(crop_mask, size, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8) * 255  # any coverage counts
        else:
            lama_mask = crop_mask
        filled = np.array(lama_model(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), Image.fromarray(lama_mask)))
        filled = cv2.cvtColor(filled[:crop.shape[0], :crop.shape[1]], cv2.COLOR_RGB2BGR)  # LaMa pads to a multiple of 8
        if scale < 1.0: filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
        selected = crop_mask > 0
        result[y1:y2, x1:x2][selected] = filled[selected]
    return result

def gpu_worker_process(gpu_id: int, image_paths: list, write_queue: mp.Queue, status_queue: mp.Queue, args: argparse.Namespace):
    """The core function executed by each GPU worker process."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch, cv2, numpy as np
    from ultralytics import YOLO
    from simple_lama_inpainting import SimpleLama
    import dino3_compat  # lets stock ultralytics load DINOv3-YOLOv12 checkpoints; no-op for plain YOLO weights
    dino3_compat.register()
    try:
        device = torch.device("cuda:0"); yolo_model = YOLO(args.weights).to(device); lama_model = SimpleLama(device=device)
    except Exception as e:
        status_queue.put({"type": "error", "message": f"GPU {gpu_id} failed to init: {e}"}); return

    for path in image_paths:
        try:
            relative_path = path.relative_to(args.input)
            output_path = args.output / relative_path
            if args.png: output_path = args.output / args.png_renames.get(relative_path.as_posix(), relative_path.with_suffix(".png").as_posix())
            img_bgr = read_image(path)
            if img_bgr is None:
                status_queue.put({"type": "log", "message": f"Could not read image: {path}"})
                status_queue.put({"type": "gpu_progress", "gpu_id": gpu_id, "result": "failed"}); continue
            predictions = yolo_model(img_bgr, conf=args.conf, verbose=False)[0]
            if len(predictions.boxes.xyxy) > 0:
                mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
                for box in predictions.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box); cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
                if args.dilate > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate, args.dilate)); mask = cv2.dilate(mask, kernel, iterations=1)
                if args.debug:
                    preview_overlay = np.zeros_like(img_bgr); preview_overlay[mask == 255] = [0, 0, 255]
                    mask_preview = cv2.addWeighted(img_bgr, 0.7, preview_overlay, 0.3, 0)
                    debug_dir = args.output / "debug" / relative_path.parent
                    # Named after the full input name, so a.jpg and a.png in one folder get separate masks.
                    write_queue.put((debug_dir / f"{path.name}_mask_raw.png", mask, None))
                    write_queue.put((debug_dir / f"{path.name}_mask_preview.png", mask_preview, None))
                rect = None
                if args.trim:
                    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                    rect = largest_clear_rect([(x, y, x + w, y + h) for x, y, w, h, _ in stats[1:]], mask.shape[1], mask.shape[0])
                    if rect and (rect[2] - rect[0]) * (rect[3] - rect[1]) < args.trim_min_keep * mask.size: rect = None
                if rect:
                    x1, y1, x2, y2 = rect
                    if path.suffix.lower() in (".jpg", ".jpeg") and not args.png: data = ("jpeg_crop", path, rect)  # the writer crops the file itself
                    else: data = img_bgr[y1:y2, x1:x2]
                    write_queue.put((output_path, data, relative_path.as_posix())); result = "trimmed"
                else:
                    result_bgr = inpaint_areas(lama_model, img_bgr, mask, args.inpaint_max_size)
                    write_queue.put((output_path, result_bgr, relative_path.as_posix())); result = "inpainted"
            elif args.skip_clean:
                result = "skipped"  # nothing to write; the main process logs the file so a resumed session skips it
            elif path.suffix.lower().lstrip('.') in LOSSY_EXTS and not args.png:
                write_queue.put((output_path, path, relative_path.as_posix())); result = "copied"
            else:
                write_queue.put((output_path, img_bgr, relative_path.as_posix())); result = "reencoded"
            status_queue.put({"type": "gpu_progress", "gpu_id": gpu_id, "result": result, "key": relative_path.as_posix()})
        except Exception as e:
            if isinstance(e, torch.OutOfMemoryError): torch.cuda.empty_cache()
            status_queue.put({"type": "error", "message": f"Error on GPU {gpu_id} processing {path.name}: {e}"})
            status_queue.put({"type": "gpu_progress", "gpu_id": gpu_id, "result": "failed"})

def cpu_writer_process(write_queue: mp.Queue, log_queue: mp.Queue, output_dir: pathlib.Path):
    """A dedicated I/O process that saves files and reports success for checkpointing."""
    import shutil
    created_dirs = set()
    while True:
        try:
            item = write_queue.get()
            if item is None: break
            path, data, log_key = item  # data is an image array to encode, a source path to copy or ("jpeg_crop", source path, rect); log_key is the input-relative path, or None for debug images
            if path.parent not in created_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
                created_dirs.add(path.parent)
            if isinstance(data, pathlib.Path): shutil.copy2(data, path); written = True
            elif isinstance(data, tuple):
                _, source, (x1, y1, x2, y2) = data
                cropped = crop_jpeg_losslessly(source, (x1, y1, x2, y2))
                if cropped is not None: path.write_bytes(cropped); written = True
                else:
                    log_queue.put(("warning", f"No lossless crop possible, re-encoded: {source}"))
                    img = read_image(source); written = img is not None and write_image(path, img[y1:y2, x1:x2])
            else: written = write_image(path, data)
            if not written: log_queue.put(("error", f"Could not encode {path}"))
            elif log_key is not None: log_queue.put(("done", log_key))
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            log_queue.put(("error", f"Could not write {path}: {e}"))
# ╰─────────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────────── Main Driver ──────────────────────────────╮
def main():
    """The main process, orchestrator of all workers and the UI."""
    console = Console()
    args = parse_cli_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.debug: (args.output / "debug").mkdir(parents=True, exist_ok=True)

    # --- Checkpoint and File Discovery ---
    log_file_path = args.output / ".processing_log.txt"
    processed_files_set = load_processed_files(log_file_path)
    if processed_files_set:
        console.print(f"[*] Found checkpoint file. Resuming session. [bold yellow]{len(processed_files_set):,}[/] files will be skipped.")

    all_image_paths = find_image_files(args.input, args.recursive, console)
    images_to_process = [p for p in all_image_paths if p.relative_to(args.input).as_posix() not in processed_files_set]

    if not images_to_process:
        console.print("[bold green]✅ All images have already been processed. Nothing to do.[/bold green]"); sys.exit(0)

    console.print(f"[*] Total images in dataset: {len(all_image_paths):,}. New images to process this session: [bold green]{len(images_to_process):,}[/bold green]")

    args.png_renames = png_output_renames([p.relative_to(args.input) for p in all_image_paths]) if args.png else {}
    if args.png_renames:
        console.print(f"[*] [yellow]{len(args.png_renames):,}[/] images share a PNG name with another image in the same folder; they keep their extension, e.g. {next(iter(args.png_renames.values()))}")

    gpu_ids = get_gpu_ids()
    if not gpu_ids: console.print("[bold red]ERROR: No NVIDIA GPUs detected.[/bold red]"); sys.exit(1)
    console.print(f"[*] Found {len(gpu_ids)} NVIDIA GPUs: {gpu_ids}")

    ctx = mp.get_context("spawn")
    write_queue = ctx.Queue(maxsize=len(gpu_ids) * 20)
    status_queue = ctx.Queue()
    log_queue = ctx.Queue()

    # --- Worker Process Setup ---
    console.print(f"[*] Starting {args.cpu_workers} CPU writer processes...")
    cpu_writers = [ctx.Process(target=cpu_writer_process, args=(write_queue, log_queue, args.output)) for _ in range(args.cpu_workers)]
    for p in cpu_writers: p.start()

    num_gpus = len(gpu_ids)
    image_slices = [images_to_process[i::num_gpus] for i in range(num_gpus)]
    console.print(f"[*] Starting {num_gpus} GPU worker processes...")
    gpu_workers = [ctx.Process(target=gpu_worker_process, args=(gpu_ids[i], image_slices[i], write_queue, status_queue, args)) for i in range(num_gpus) if image_slices[i]]
    for p in gpu_workers: p.start()

    processing_start_time = time.time()

    # --- Live Display and Main Monitoring Loop ---
    progress = Progress(
        TextColumn("[bold blue]Processing..."), BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%", "•",
        TextColumn("[cyan]{task.completed} of {task.total}"), "•",
        SpeedColumn(), "•", EstimatedTimeRemainingColumn(), console=console
    )
    progress_task = progress.add_task("New Images", total=len(images_to_process))
    RESULTS = ("inpainted", "trimmed", "skipped", "copied", "reencoded", "failed")
    gpu_stats = {gpu_id: {'completed': 0, **dict.fromkeys(RESULTS, 0), 'start_time': time.time()} for gpu_id in gpu_ids}
    clean_label = "skipped" if args.skip_clean else "written"

    def generate_layout():
        gpu_table = Table.grid(expand=True)
        gpu_table.add_column("GPU ID", justify="right", style="cyan", no_wrap=True)
        gpu_table.add_column("Processed", justify="center", style="magenta")
        gpu_table.add_column("Watermarked", justify="center", style="red")
        gpu_table.add_column("No watermark", justify="center", style="yellow")
        gpu_table.add_column("Speed (img/s)", justify="center", style="green")
        for gpu_id, stats in gpu_stats.items():
            elapsed = time.time() - stats['start_time']
            rate = stats['completed'] / elapsed if elapsed > 1 else 0.0
            clean = stats['skipped'] + stats['copied'] + stats['reencoded']
            marked = f"{stats['trimmed']:,} trimmed, {stats['inpainted']:,} inpainted" if args.trim else f"{stats['inpainted']:,} inpainted"
            gpu_table.add_row(f"[bold]GPU {gpu_id}[/]", f"{stats['completed']:,} images", marked, f"{clean:,} {clean_label}", f"{rate:.2f} img/s")

        display_grid = Table.grid(padding=(0,0,1,0))
        display_grid.add_row(Panel(gpu_table, title="[bold]GPU Worker Status[/bold]", border_style="green"))
        display_grid.add_row(progress)
        return Panel(display_grid, title=f"[bold]Processing Session[/bold] ([yellow]{len(processed_files_set):,}[/] previously completed)", border_style="blue")

    log_file = open(log_file_path, "a", encoding="utf-8")

    # The live display runs on the alternate screen, which is discarded on exit, so problems are also kept for the summary.
    problems = []
    def report(message: str, live_console=True):
        problems.append(message)
        if live_console: console.log(message)

    def drain_log_queue():
        """Appends every file the CPU writers have finished to the checkpoint log."""
        try:
            while True:
                kind, value = log_queue.get_nowait()
                if kind == "done": log_file.write(f"{value}\n")
                elif kind == "warning": report(f"⚠️ [yellow]WARNING:[/] {value}", live_console=False)
                else: report(f"❌ [bold red]WRITE ERROR:[/bold red] {value}", live_console=False)
        except queue.Empty:
            pass
        log_file.flush()

    total_completed = 0
    interrupted = False
    try:
        with Live(generate_layout(), console=console, screen=True, redirect_stderr=False, vertical_overflow="crop") as live:
            while total_completed < len(images_to_process):
                # Checked before draining: a worker that has exited has already flushed its last messages into the queue.
                gpu_workers_done = not any(p.is_alive() for p in gpu_workers)
                try:
                    while True: # Process all available status messages
                        msg = status_queue.get_nowait()
                        if msg["type"] == "gpu_progress":
                            gpu_stats[msg["gpu_id"]]['completed'] += 1
                            gpu_stats[msg["gpu_id"]][msg["result"]] += 1
                            total_completed += 1
                            if msg["result"] == "skipped": log_file.write(f"{msg['key']}\n")
                        elif msg["type"] == "error": report(f"❌ [bold red]ERROR:[/bold red] {msg['message']}")
                        elif msg["type"] == "log": report(f"⚠️ [yellow]WARNING:[/] {msg['message']}")
                except queue.Empty:
                    pass

                drain_log_queue()
                progress.update(progress_task, completed=total_completed)
                live.update(generate_layout())
                if gpu_workers_done: break  # e.g. a GPU failed to initialize; its images are left for the next session
                time.sleep(0.1)

    except KeyboardInterrupt:
        interrupted = True
        console.print("\n[!] Pausing session... Please wait for checkpointing to complete.", style="bold yellow")

    # --- Clean Shutdown Sequence ---
    # GPU workers are finished (or interrupted); the CPU writers still hold up to one full write queue.
    # Each writer stops at its None sentinel, so every image queued before it is written and logged.
    console.print("\n[*] Finishing queued writes and shutting down all workers...")
    for p in gpu_workers:
        if interrupted and p.is_alive(): p.terminate()
        p.join(timeout=30)
        if p.is_alive(): p.terminate()
    for _ in cpu_writers:
        try: write_queue.put(None, timeout=5)
        except queue.Full: break  # writers already stopped by Ctrl+C; the fallback below terminates them
    deadline = time.time() + (10 if interrupted else 300)
    while any(p.is_alive() for p in cpu_writers) and time.time() < deadline:
        drain_log_queue()  # keeps the log pipe empty, so no writer blocks on exit
        for p in cpu_writers: p.join(timeout=0.05)
    for p in cpu_writers:
        if p.is_alive(): p.terminate(); p.join(timeout=5)
    drain_log_queue()
    log_file.close()
    write_queue.cancel_join_thread()  # a sentinel left in a full queue must not block exit

    duration_seconds = time.time() - processing_start_time
    minutes, seconds = divmod(duration_seconds, 60)

    final = {key: sum(stats[key] for stats in gpu_stats.values()) for key in ('completed',) + RESULTS}
    clean_count = final['skipped'] + final['copied'] + final['reencoded']

    console.print("\n[--- [bold yellow]Session Paused / Complete[/bold yellow] ---]")
    console.print(f"✅ Processed [bold]{final['completed']:,}[/] new images this session in [bold]{int(minutes)} minutes and {seconds:.2f} seconds[/bold].")
    if args.trim: console.print(f"   {final['trimmed'] + final['inpainted']:,} had a watermark: {final['trimmed']:,} were trimmed, {final['inpainted']:,} inpainted (watermark too far from an edge).")
    else: console.print(f"   {final['inpainted']:,} had a watermark and were inpainted.")
    if args.skip_clean: console.print(f"   {clean_count:,} had no detected watermark and were not written.")
    else: console.print(f"   {clean_count:,} had no detected watermark: {final['copied']:,} copied unchanged, {final['reencoded']:,} re-encoded.")
    if final['failed']: console.print(f"   [bold red]{final['failed']:,} failed[/] and will be retried in the next session.")
    if problems:
        console.print(f"   [bold red]{len(problems):,} problems[/] were reported; files that were not written will be retried in the next session:")
        for message in problems[:20]: console.print(f"     {message}")
        if len(problems) > 20: console.print(f"     ... and {len(problems) - 20:,} more")
    console.print(f"✅ Clean images are saved in: [link=file://{args.output.resolve()}]{args.output.resolve()}[/link]")

# ╰─────────────────────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    main()
