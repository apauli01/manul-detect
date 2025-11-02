#!/usr/bin/env python3
"""infer_gui.py

Simple Tkinter GUI to step through images in a folder and run YOLOv8 inference on them.

Usage:
  python infer_gui.py --folder /path/to/test/images --model runs/train/train3/weights/best.pt

Controls:
  - Next / Prev: move through images (no inference)
  - Infer: run model on current image and overlay boxes
  - Save: save annotated image to output folder (default: ./inference_out)

Notes:
  - Requires `ultralytics` and `Pillow` installed in the same environment.
  - GUI uses tkinter (part of stdlib) and Pillow's ImageTk.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import List, Tuple

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    from PIL import Image, ImageTk, ImageDraw, ImageFont
except Exception as e:
    print("Tkinter/Pillow required for GUI. Install Pillow and run in a desktop environment.")
    raise

LOG = logging.getLogger("infer_gui")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _get_text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    """Robust text size extraction across Pillow versions."""
    try:
        # Pillow >= 8.0: textbbox is available
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    except Exception:
        try:
            return font.getsize(text)
        except Exception:
            try:
                return draw.textsize(text, font=font)
            except Exception:
                # fallback estimate
                return (len(text) * 7, 12)


def list_images(folder: Path) -> List[Path]:
    imgs = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXTS]
    return imgs


class YOLOModelWrapper:
    def __init__(self, model_path: str, device: str | None = None):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            LOG.error("Failed to import ultralytics: %s", exc)
            raise
        self.model = YOLO(model_path)
        self.names = getattr(self.model, 'names', {})
        self.device = device

    def predict(self, img_path: Path, imgsz: int = 640, conf: float = 0.25):
        # returns list of (xyxy, cls, conf)
        start = time.time()
        res = self.model.predict(source=str(img_path), imgsz=imgsz, conf=conf, device=self.device, verbose=False)
        stop = time.time()
        ms = (stop - start) * 1000.0
        LOG.info("Inference on %s took %.1f ms", img_path.name, ms)
        if not res:
            return []
        r = res[0]
        boxes = []
        # boxes stored in r.boxes
        bobj = getattr(r, 'boxes', None)
        if bobj is None:
            return []
        # try different attributes
        data = getattr(bobj, 'data', None)
        if data is not None:
            # tensor-like; convert to list
            try:
                arr = data.cpu().numpy()
                for row in arr:
                    # Ultralytics typically stores [x1, y1, x2, y2, conf, cls]
                    x1, y1, x2, y2, confv, cls = row[:6]
                    boxes.append(((x1, y1, x2, y2), int(cls), float(confv)))
                return boxes
            except Exception:
                pass
        # fallback: use xyxy and cls
        try:
            xyxy = getattr(bobj, 'xyxy', None)
            clslist = getattr(bobj, 'cls', None)
            confs = getattr(bobj, 'conf', None)
            if xyxy is not None:
                xy = xyxy.cpu().numpy() if hasattr(xyxy, 'cpu') else xyxy
                for i, coords in enumerate(xy):
                    x1, y1, x2, y2 = coords[:4]
                    confv = float(confs[i]) if confs is not None else 0.0
                    cls = int(clslist[i]) if clslist is not None else 0
                    boxes.append(((x1, y1, x2, y2), cls, confv))
                return boxes
        except Exception:
            pass
        return []


class InferenceGUI:
    def __init__(self, folder: Path, model_path: str, imgsz: int = 640, conf: float = 0.25, device: str | None = None, outdir: Path | None = None):
        self.folder = folder
        self.imgs = list_images(folder)
        if not self.imgs:
            raise SystemExit(f"No images found in {folder}")
        self.index = 0
        self.imgsz = imgsz
        self.conf = conf
        self.outdir = outdir or (Path.cwd() / "inference_out")
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.model = None
        try:
            self.model = YOLOModelWrapper(model_path, device=device)
        except Exception as exc:
            messagebox.showerror("Model load error", f"Failed to load model: {exc}")
            raise

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title("YOLO Inference GUI")
        self._build_ui()
        self.show_image()

    def _build_ui(self):
        frm = ttk.Frame(self.root)
        frm.pack(fill=tk.BOTH, expand=True)

        # Canvas for image
        self.canvas = tk.Canvas(frm, width=800, height=600, bg='black')
        self.canvas.grid(row=0, column=0, rowspan=6, sticky='nsew')

        # Controls
        ttk.Button(frm, text='Prev', command=self.prev_image).grid(row=0, column=1, sticky='ew')
        ttk.Button(frm, text='Next', command=self.next_image).grid(row=1, column=1, sticky='ew')
        ttk.Button(frm, text='Infer', command=self.infer_current).grid(row=2, column=1, sticky='ew')
        ttk.Button(frm, text='Save Annotated', command=self.save_annotated).grid(row=3, column=1, sticky='ew')
        ttk.Button(frm, text='Quit', command=self.root.quit).grid(row=5, column=1, sticky='ew')

        self.info = tk.StringVar()
        ttk.Label(frm, textvariable=self.info, wraplength=200).grid(row=4, column=1, sticky='n')

        frm.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)

    def show_image(self, annotated: Image.Image | None = None):
        img_path = self.imgs[self.index]
        if annotated is None:
            pil = Image.open(img_path).convert('RGB')
        else:
            pil = annotated

        # resize to fit canvas
        # ensure geometry info is up to date
        try:
            self.root.update_idletasks()
        except Exception:
            pass
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        # fallback to defaults if widget not yet laid out
        if not cw or cw < 10:
            cw = 800
        if not ch or ch < 10:
            ch = 600
        w, h = pil.size
        scale = min(cw / w, ch / h, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        disp = pil.resize((nw, nh), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.delete('all')
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo, anchor='center')
        self.info.set(f"{img_path.name}  ({self.index+1}/{len(self.imgs)})")

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_image()

    def next_image(self):
        if self.index < len(self.imgs) - 1:
            self.index += 1
            self.show_image()

    def infer_current(self):
        img_path = self.imgs[self.index]
        # run in thread to keep UI responsive
        threading.Thread(target=self._run_infer, args=(img_path,), daemon=True).start()

    def _run_infer(self, img_path: Path):
        self.info.set(f"Running inference on {img_path.name}...")
        boxes = []
        try:
            boxes = self.model.predict(img_path, imgsz=self.imgsz, conf=self.conf)
        except Exception as exc:
            LOG.exception("Inference failed: %s", exc)
            messagebox.showerror("Inference error", str(exc))
            return
        # open original image and draw boxes
        im = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        for (x1, y1, x2, y2), cls, conf in boxes:
            # draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            name = self.model.names.get(cls, str(cls))
            text = f"{name} {conf:.2f}"
            text_w, text_h = _get_text_size(draw, text, font)
            # ensure integer coords
            tx1 = int(x1)
            ty1 = int(y1 - text_h - 4)
            tx2 = int(x1 + text_w + 4)
            ty2 = int(y1)
            draw.rectangle([tx1, ty1, tx2, ty2], fill='red')
            draw.text((x1 + 2, y1 - text_h - 2), text, fill='white', font=font)

        # show annotated
        self.annotated_image = im
        self.show_image(annotated=im)

    def save_annotated(self):
        if not hasattr(self, 'annotated_image') or self.annotated_image is None:
            messagebox.showinfo("Save", "No annotated image to save. Run inference first.")
            return
        img_path = self.imgs[self.index]
        outp = self.outdir / f"{img_path.stem}_annotated{img_path.suffix}"
        try:
            self.annotated_image.save(outp)
            messagebox.showinfo("Saved", f"Saved annotated image to {outp}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    def run(self):
        self.root.mainloop()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Simple YOLOv8 inference GUI')
    p.add_argument('--folder', required=True, help='Folder with test images')
    p.add_argument('--model', required=True, help='YOLO model path (weights)')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--device', default=None, help='Device for ultralytics (e.g. 0 or cpu)')
    p.add_argument('--outdir', default='./inference_out', help='Where to save annotated images')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}")
        return 2

    gui = InferenceGUI(folder, args.model, imgsz=args.imgsz, conf=args.conf, device=args.device, outdir=Path(args.outdir))
    gui.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
