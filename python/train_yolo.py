#!/usr/bin/env python3
"""train_yolo.py

Train a YOLO model on a dataset in YOLO format (one .txt label per image with
: class x_center y_center width height normalized) using the Ultralytics
package (YOLOv8). This script will:

- Scan a provided dataset folder (images + labels) to discover classes.
- Generate a temporary data YAML file required by the Ultralytics trainer.
- Invoke the Ultralytics `YOLO` training API with configurable options.

Assumptions / reasonable inferences:
- Your dataset root contains two subfolders: `images` and `labels`, or you can
  provide `--train-list` and `--val-list` files listing image paths.
- Label files follow standard YOLO text format.

If you want an older YOLO v1/v2-specific trainer, provide details and I can
adapt; otherwise YOLOv8 will train on the same label format and is easier to
install and run.

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
try:
    from PIL import Image, UnidentifiedImageError
except Exception:
    Image = None
    UnidentifiedImageError = None
try:
    # SummaryWriter is optional; prefer torch's one if available
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    try:
        from tensorboard.summary.writer.event_file_writer import EventFileWriter as SummaryWriter  # type: ignore
    except Exception:
        SummaryWriter = None
import csv
try:
    import numpy as np
except Exception:
    np = None

LOG = logging.getLogger("train_yolo")


def find_images_and_labels(root: Path) -> Tuple[List[Path], List[Path]]:
    """Search for image files and matching label files under `root`.

    This function expects images under a directory `images/` (recursively) and
    label files under `labels/` with the same base filename but .txt extension.
    It also accepts a flat structure where images and labels are together.
    """
    imgs = []
    labs = []
    # common image extensions
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # search for images recursively
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in img_exts:
            imgs.append(p)
            # corresponding label path
            label = p.with_suffix(":.txt")  # placeholder
    # If no images found under root, try images/ and labels/ subfolders
    if not imgs:
        images_dir = root / "images"
        labels_dir = root / "labels"
        if images_dir.exists():
            for p in images_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in img_exts:
                    imgs.append(p)
        else:
            LOG.warning("No images found in %s and no `images/` dir present", root)
        if labels_dir.exists():
            for p in labels_dir.rglob("*.txt"):
                labs.append(p)
        else:
            LOG.warning("No `labels/` dir present under %s", root)
        # try to pair images and labels by basename
        paired_imgs = []
        paired_labs = []
        labs_map = {p.stem: p for p in labs}
        for img in imgs:
            lab = labels_dir / (img.stem + ".txt")
            if lab.exists():
                paired_imgs.append(img)
                paired_labs.append(lab)
        return paired_imgs, paired_labs

    # For the recursive case we try to find labels next to each image with .txt
    paired_imgs = []
    paired_labs = []
    for img in imgs:
        lab = img.with_suffix(".txt")
        if lab.exists():
            paired_imgs.append(img)
            paired_labs.append(lab)
        else:
            # sometimes labels are in labels/ with same stem
            lab2 = img.parent.parent / "labels" / (img.stem + ".txt")
            if lab2.exists():
                paired_imgs.append(img)
                paired_labs.append(lab2)
            else:
                LOG.debug("No label for image %s", img)
    return paired_imgs, paired_labs


def discover_class_names(label_paths: List[Path]) -> Dict[int, str]:
    """Return a mapping of class id -> class name. If names are not present,
    use numeric names ("0", "1", ...)."""
    class_ids = set()
    for p in label_paths:
        try:
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                cls = int(float(parts[0]))
                class_ids.add(cls)
        except Exception as exc:
            LOG.warning("Failed reading label %s: %s", p, exc)
    if not class_ids:
        LOG.warning("No classes discovered in label files")
        return {}
    # build names as string of the id by default
    names = {i: str(i) for i in sorted(class_ids)}
    return names


def parse_darknet_data_file(path: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Parse a Darknet-style .data file and return (train_path, val_path, names_path).

    Returns None for any field not found.
    """
    train = None
    val = None
    names = None
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("train="):
                t = line.split("=", 1)[1].strip()
                train = Path(t) if os.path.isabs(t) else (path.parent / t)
            elif line.startswith("valid=") or line.startswith("val="):
                v = line.split("=", 1)[1].strip()
                val = Path(v) if os.path.isabs(v) else (path.parent / v)
            elif line.startswith("names="):
                n = line.split("=", 1)[1].strip()
                names = Path(n) if os.path.isabs(n) else (path.parent / n)
    except Exception as exc:
        LOG.warning("Failed parsing darknet data file %s: %s", path, exc)
    return train, val, names


def normalize_list_file(list_path: Path, tmpdir: Path, dataset_root: Path) -> Path:
    """Read a train/val list file and write a normalized copy (absolute paths) into tmpdir.

    Heuristics applied when a listed path doesn't exist:
    - If it starts with 'data/', strip that prefix and try again relative to the dataset_root.
    - If still not found, try resolving by basename under dataset_root (recursive search limited).
    Returns path to the new list file in tmpdir (or the original if already absolute and seems valid).
    """
    out = tmpdir / list_path.name
    lines = []
    try:
        raw = list_path.read_text().splitlines()
    except Exception:
        LOG.warning("Could not read list file %s", list_path)
        return list_path

    for ln in raw:
        s = ln.strip()
        if not s:
            continue
        p = Path(s)
        if not p.is_absolute():
            # first, resolve relative to the list file's directory
            p = (list_path.parent / p).resolve()
        if not p.exists():
            # heuristic: strip leading 'data/' if present and try relative to dataset_root
            sp = s
            if sp.startswith("data/"):
                candidate = (dataset_root / sp[len("data/"):]).resolve()
                if candidate.exists():
                    p = candidate
            # heuristic: try find by basename under dataset_root
            if not p.exists():
                basename = Path(s).name
                found = None
                for cand in dataset_root.rglob(basename):
                    if cand.is_file():
                        found = cand
                        break
                if found:
                    p = found
        lines.append(str(p.resolve()))

    out.write_text("\n".join(lines))
    return out


def read_names_file(names_path: Path) -> Dict[int, str]:
    try:
        lines = [l.strip() for l in names_path.read_text().splitlines() if l.strip()]
        return {i: name for i, name in enumerate(lines)}
    except Exception as exc:
        LOG.warning("Failed reading names file %s: %s", names_path, exc)
        return {}


def write_data_yaml(train_paths: List[Path], val_paths: List[Path], names: Dict[int, str], out_path: Path) -> None:
    # Ensure all paths are absolute. If train_paths/val_paths contain a single
    # .txt file, write its absolute path (Ultralytics expects valid paths).
    def prepare(paths: List[Path]):
        if len(paths) == 1 and paths[0].suffix.lower() == ".txt":
            return str(paths[0].resolve())
        return [str(p.resolve()) for p in paths]

    data = {
        "train": prepare(train_paths),
        "val": prepare(val_paths),
        "nc": len(names),
        "names": names,
    }
    out_path.write_text(json.dumps(data, indent=2))
    LOG.info("Wrote data yaml to %s", out_path)


def make_train_val_lists(all_images: List[Path], val_ratio: float = 0.2) -> Tuple[List[Path], List[Path]]:
    from random import shuffle

    imgs = list(all_images)
    shuffle(imgs)
    cut = int(len(imgs) * (1 - val_ratio))
    return imgs[:cut], imgs[cut:]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO (Ultralytics) on YOLO-format dataset")
    p.add_argument("--dataset", required=True, help="Path to dataset root (images/ and labels/ or flat structure)")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio if no val set provided")
    p.add_argument("--model", default="yolov8n.pt", help="Pretrained model to use or model name (e.g., yolov8n.pt)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--project", default="runs/train", help="Where to save results")
    p.add_argument("--name", default=None, help="Experiment name")
    p.add_argument("--device", default=None, help="Device like 0 or cpu")
    p.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logger (Ultralytics) and suggest running tensorboard pointing to the project dir")
    p.add_argument("--data-darknet", default=None, help="Path to Darknet obj.data file (optional)")
    p.add_argument("--names", default=None, help="Path to obj.names file (optional)")
    p.add_argument("--dry-run", action="store_true", help="Validate dataset files and labels, then exit without training")
    p.add_argument("--no-augment", dest="augment", action="store_false", help="Disable augmentation")
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def validate_dataset_entry_list(list_path: Path, max_errors: int = 20) -> Tuple[int, List[str]]:
    """Validate a train/val list file containing one image path per line.

    Returns (num_valid, errors)
    """
    errors: List[str] = []
    valid = 0
    if not list_path.exists():
        errors.append(f"List file not found: {list_path}")
        return 0, errors
    for i, line in enumerate(list_path.read_text().splitlines()):
        if not line.strip():
            continue
        img_path = Path(line.strip())
        # resolve relative paths relative to the list file
        if not img_path.is_absolute():
            img_path = (list_path.parent / img_path).resolve()
        if not img_path.exists():
            errors.append(f"Missing image: {img_path}")
        else:
            # try opening image to detect corruption
            if Image is not None:
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except UnidentifiedImageError:
                    errors.append(f"Corrupt or unsupported image: {img_path}")
                except Exception as exc:
                    errors.append(f"Failed to read image {img_path}: {exc}")
            # check label file exists next to image
            lab = img_path.with_suffix('.txt')
            if not lab.exists():
                # also check labels/imagename.txt
                alt = img_path.parent / 'labels' / (img_path.stem + '.txt')
                if not alt.exists():
                    errors.append(f"Missing label for image: {img_path} (checked: {lab}, {alt})")
                else:
                    # replace lab for format checking
                    lab = alt
            # basic label format check
            if lab.exists():
                try:
                    for ln in lab.read_text().splitlines():
                        s = ln.strip()
                        if not s:
                            continue
                        parts = s.split()
                        if len(parts) < 5:
                            errors.append(f"Bad label format in {lab}: '{s}'")
                            break
                        # first should be int
                        try:
                            int(float(parts[0]))
                        except Exception:
                            errors.append(f"Non-integer class id in {lab}: '{parts[0]}'")
                            break
                except Exception as exc:
                    errors.append(f"Failed reading label {lab}: {exc}")
        if len(errors) >= max_errors:
            break
    # count valid as total lines minus errors (approx)
    try:
        total_lines = len([l for l in list_path.read_text().splitlines() if l.strip()])
    except Exception:
        total_lines = 0
    valid = max(0, total_lines - len(errors))
    return valid, errors


def validate_and_report(train_list: Path, val_list: Path) -> bool:
    """Run validation on train and val lists. Returns True if OK, False if problems found."""
    all_errors: List[str] = []
    for label, lst in ("train", train_list), ("val", val_list):
        valid, errors = validate_dataset_entry_list(lst)
        if errors:
            all_errors.append(f"Problems in {label} list ({lst}):")
            all_errors.extend(errors[:50])
        else:
            LOG.info("%s list looks OK: %s entries", label, valid)

    # check cache directory writability (Ultralytics may create a cache dir next to lists)
    try:
        cache_dir = train_list.parent
        test_file = cache_dir / ".ultralytics_write_test"
        with open(test_file, "w") as f:
            f.write("test")
        test_file.unlink()
    except Exception:
        all_errors.append(f"Cache directory {train_list.parent} is not writeable. Ultralytics may not save cache.")

    if all_errors:
        LOG.error("Dataset validation found issues:\n%s", "\n".join(all_errors[:200]))
        return False
    return True


def write_tensorboard_from_results(run_dir: Path) -> None:
    """Create TensorBoard events under run_dir from results.csv and example images.

    This is a fallback for when Ultralytics didn't create event files. It writes
    scalar metrics from results.csv (per-epoch) and a few example images.
    """
    if SummaryWriter is None:
        LOG.warning("No SummaryWriter available (install torch or tensorboard); cannot write event files")
        return

    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        LOG.info("No results.csv at %s; nothing to write for TensorBoard", results_csv)
        return

    writer = SummaryWriter(log_dir=str(run_dir))
    try:
        with open(results_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # try to pick a step index
                step = None
                if 'epoch' in row and row['epoch']:
                    try:
                        step = int(float(row['epoch']))
                    except Exception:
                        step = None
                # write all numeric columns as scalars
                for k, v in row.items():
                    if v is None or v == '':
                        continue
                    try:
                        val = float(v)
                    except Exception:
                        continue
                    tag = k.strip()
                    if step is not None:
                        try:
                            writer.add_scalar(tag, val, step)
                        except Exception:
                            pass
                    else:
                        try:
                            # use incremental global_step
                            writer.add_scalar(tag, val)
                        except Exception:
                            pass

        # add a few example images if numpy & PIL available
        if Image is not None and np is not None:
            imgs = list(run_dir.glob('train_batch*.jpg')) + list(run_dir.glob('val_batch0_pred.jpg'))
            for i, imgp in enumerate(imgs[:5]):
                try:
                    with Image.open(imgp) as im:
                        arr = np.array(im.convert('RGB'))
                        # writer expects HWC by default
                        writer.add_image(f'example/{imgp.name}', arr, dataformats='HWC')
                except Exception:
                    continue
    finally:
        try:
            writer.close()
        except Exception:
            pass


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(args.dataset)
    if not root.exists():
        LOG.error("Dataset path does not exist: %s", root)
        return 2
    # Prefer existing Darknet-style files if present
    train_list = None
    val_list = None
    names = {}

    # 1) If user supplied a darknet data file, parse it
    if args.data_darknet:
        dd = Path(args.data_darknet)
        if dd.exists():
            t, v, n = parse_darknet_data_file(dd)
            if t:
                train_list = t
            if v:
                val_list = v
            if n and n.exists():
                names = read_names_file(n)

    # 2) If Train.txt / Validation.txt exist under dataset root, use them
    train_txt_candidate = root / "Train.txt"
    val_txt_candidate = root / "Validation.txt"
    if train_txt_candidate.exists():
        train_list = train_txt_candidate.resolve()
    if val_txt_candidate.exists():
        val_list = val_txt_candidate.resolve()

    # 3) If obj.names provided or present
    names_file = None
    if args.names:
        names_file = Path(args.names)
    else:
        cand = root / "obj.names"
        if cand.exists():
            names_file = cand
    if names_file and names_file.exists():
        names = read_names_file(names_file)

    # 4) If we still don't have train/val lists, look for obj_Train_data / obj_Validation_data
    tmpdir = None
    if not train_list or not val_list:
        train_dir = root / "obj_Train_data"
        val_dir = root / "obj_Validation_data"
        if train_dir.exists() and val_dir.exists():
            tmpdir = Path(tempfile.mkdtemp(prefix="yolodata_"))
            train_list = tmpdir / "train.txt"
            val_list = tmpdir / "val.txt"
            def gather_images(d: Path):
                exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
                return [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts]
            train_imgs = gather_images(train_dir)
            val_imgs = gather_images(val_dir)
            train_list.write_text("\n".join(str(p.resolve()) for p in train_imgs))
            val_list.write_text("\n".join(str(p.resolve()) for p in val_imgs))

    # If we still don't have lists, fallback to scan of dataset root
    if not train_list or not val_list:
        imgs, labs = find_images_and_labels(root)
        if not imgs:
            LOG.error("No images found in dataset root: %s", root)
            return 2
        if not labs:
            LOG.error("No label files found - ensure YOLO-format labels exist next to images or under labels/")
            return 2
        LOG.info("Found %s images with %s label files", len(imgs), len(labs))
        if not names:
            names = discover_class_names(labs)
        if not names:
            LOG.error("No classes discovered; aborting")
            return 2
        train_imgs, val_imgs = make_train_val_lists(imgs, args.val_ratio)
        if tmpdir is None:
            tmpdir = Path(tempfile.mkdtemp(prefix="yolodata_"))
        train_list = tmpdir / "train.txt"
        val_list = tmpdir / "val.txt"
        train_list.write_text("\n".join(str(p) for p in train_imgs))
        val_list.write_text("\n".join(str(p) for p in val_imgs))

    # If names are still empty, try discovering from label files listed in train_list
    if not names:
        try:
            # read label files from train_list
            label_paths = []
            for line in Path(train_list).read_text().splitlines():
                img = Path(line.strip())
                lab = img.with_suffix('.txt')
                if lab.exists():
                    label_paths.append(lab)
            names = discover_class_names(label_paths)
        except Exception:
            names = {}

    if not names:
        LOG.error("No class names discovered; please provide obj.names or a names file")
        return 2

    LOG.info("Using train list: %s", train_list)
    LOG.info("Using val list: %s", val_list)

    # create a temporary dir to hold normalized list files and the data.yaml
    data_tmpdir = Path(tempfile.mkdtemp(prefix="yolodata_"))

    # Normalize any existing list files to absolute paths and write them into data_tmpdir
    try:
        if train_list and Path(train_list).exists():
            train_list = normalize_list_file(Path(train_list), data_tmpdir, root)
        if val_list and Path(val_list).exists():
            val_list = normalize_list_file(Path(val_list), data_tmpdir, root)
    except Exception as exc:
        LOG.warning("Failed to normalize list files: %s", exc)

    # write data yaml
    data_yaml_path = data_tmpdir / "data.yaml"
    write_data_yaml([train_list], [val_list], names, data_yaml_path)

    # try to import ultralytics
    try:
        from ultralytics import YOLO
    except Exception as exc:
        LOG.error("Failed importing ultralytics: %s\nInstall with: pip install ultralytics", exc)
        return 3

    # determine device: prefer args.device, else GPU if available
    device = args.device
    if device is None:
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "0"

    # configure training
    yolo = YOLO(args.model)
    train_kwargs = {
        "data": str(data_yaml_path),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.img_size,
        "project": args.project,
        "device": device,
    }
    if args.name:
        train_kwargs["name"] = args.name
    if not args.augment:
        train_kwargs["augment"] = False
    # Ensure we train for detection task (bounding boxes)
    train_kwargs["task"] = "detect"

    LOG.info("Starting training with model=%s epochs=%s imgsz=%s batch=%s device=%s", args.model, args.epochs, args.img_size, args.batch, device)

    # Validate dataset before training if requested or always warn
    if args.dry_run:
        ok = validate_and_report(Path(train_list), Path(val_list))
        if not ok:
            LOG.error("Dry-run: dataset validation failed. Fix issues and re-run.")
            return 4
        LOG.info("Dry-run: dataset validation passed. Exiting without training as requested.")
        return 0

    # final validation (non-fatal): warn and continue or abort based on severity
    ok = validate_and_report(Path(train_list), Path(val_list))
    if not ok:
        LOG.warning("Dataset validation reported problems; training may fail. You can run with --dry-run to just validate and exit.")

    # start training
    yolo.train(**train_kwargs)

    # After training, if user requested tensorboard, ensure there are event files.
    if args.tensorboard:
        # find the latest run directory under project
        proj = Path(args.project)
        if proj.exists():
            try:
                subdirs = [d for d in proj.iterdir() if d.is_dir()]
                if subdirs:
                    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
                else:
                    latest = proj
            except Exception:
                latest = proj
        else:
            latest = proj

        # check for existing event files
        events = list(latest.rglob('events.out.tfevents*'))
        if events:
            LOG.info("TensorBoard event files found under %s", latest)
        else:
            LOG.info("No TensorBoard event files found under %s; attempting to write events from results.csv", latest)
            try:
                write_tensorboard_from_results(latest)
            except Exception as exc:
                LOG.warning("Failed to write TensorBoard events: %s", exc)

        LOG.info("Run TensorBoard: tensorboard --logdir %s", args.project)

    # cleanup temporary directories if we created them
    try:
        if tmpdir:
            shutil.rmtree(tmpdir)
        # data_yaml_path parent is a tmpdir we created above
        try:
            shutil.rmtree(data_yaml_path.parent)
        except Exception:
            pass
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
