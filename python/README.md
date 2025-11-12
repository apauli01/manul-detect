# ManulDetect (YOLO training & inference)

Lightweight utilities to train and run YOLOv8 models on a YOLO-format dataset (one .txt label per image). The repository includes scripts for preparing datasets, training with Ultralytics YOLO, and a simple Tkinter GUI for running inference and saving annotated images.

## Contents
- Training script: [python/train_yolo.py](python/train_yolo.py) (`train_yolo.main`, `train_yolo.parse_args`)  
- Inference GUI: [python/infer_gui.py](python/infer_gui.py) (`infer_gui.InferenceGUI`, `infer_gui.parse_args`)  
- Dataset samples: [python/trainingData/manuldetect/Train.txt](python/trainingData/manuldetect/Train.txt), [python/trainingData/manuldetect/Validation.txt](python/trainingData/manuldetect/Validation.txt)  
- Example weights: [python/yolov8n.pt](python/yolov8n.pt), [python/yolo11n.pt](python/yolo11n.pt)  
- Requirements: [python/requirements.txt](python/requirements.txt)  
- Output runs: [python/runs/train/](python/runs/train/)  
- Image folders: [python/images10000/](python/images10000/), [python/images8000/](python/images8000/)

## Requirements
Install the Python requirements listed in [python/requirements.txt](python/requirements.txt). At minimum you will need:
- ultralytics (YOLOv8)
- Pillow (for the GUI)
- torch (if using GPU)

Example:
```sh
# install dependencies
pip install -r python/requirements.txt
```

## Dataset layout
The trainer accepts:
- A dataset root containing YOLO-style image/label pairs (labels next to images with `.txt`), OR
- Pre-made text lists (`Train.txt` / `Validation.txt`) that list image paths, OR
- `obj_Train_data` and `obj_Validation_data` folders (the script will generate temporary lists).

See [python/train_yolo.py](python/train_yolo.py) for detection of these formats and dataset handling (functions such as `train_yolo.make_train_val_lists` and `train_yolo.write_data_yaml`).

If you already have lists, place them as:
- [python/trainingData/manuldetect/Train.txt](python/trainingData/manuldetect/Train.txt)  
- [python/trainingData/manuldetect/Validation.txt](python/trainingData/manuldetect/Validation.txt)

## Training
Run the main trainer script with the dataset root. Typical usage:
```sh
python python/train_yolo.py --dataset python/trainingData/manuldetect --model yolov8n.pt --epochs 50 --batch 16
```
Key options (see `train_yolo.parse_args`):
- `--dataset` : dataset root or folder containing `Train.txt`/`Validation.txt`
- `--model` : pretrained weights (e.g., `yolov8n.pt`)
- `--epochs`, `--batch`, `--img-size`
- `--project` : output project dir (default `runs/train`)
- `--dry-run` : validate dataset then exit (uses `train_yolo.validate_and_report`)

The script will:
- Discover images and labels, attempt to infer classes if not provided, write a `data.yaml` using `train_yolo.write_data_yaml`, and call Ultralytics `YOLO(...).train(...)`.

If you need a specific train/val split and there are no lists, the script uses `train_yolo.make_train_val_lists` to split images.

## Inference (GUI)
A small Tkinter-based GUI allows stepping through images and running inference:
```sh
python python/infer_gui.py --folder path/to/images --model runs/train/exp/weights/best.pt --outdir ./inference_out
```
See [python/infer_gui.py](python/infer_gui.py) (`infer_gui.InferenceGUI`) for controls:
- Next / Prev: browse images
- Infer: run model and draw boxes
- Save: save annotated image to `--outdir`

## Inspecting results / TensorBoard
Training outputs are saved under the `--project` directory (default `python/runs/train/`). If you enable tensorboard (`--tensorboard`) or want to inspect results, run:
```sh
tensorboard --logdir python/runs/train
```
The trainer tries to write TensorBoard events from CSV if native event files are missing.

## Tips
- Use `--dry-run` to validate dataset files/labels before committing to long runs. (`train_yolo.validate_and_report`)
- Provide `--names` or an `obj.names` file in the dataset root to control class names.
- If training on GPU, set `--device 0` (or leave unset; the script will attempt to auto-detect).

## Where to look in the code
- Core training logic and CLI: [python/train_yolo.py](python/train_yolo.py) (`train_yolo.main`, `train_yolo.parse_args`)  
- Inference GUI: [python/infer_gui.py](python/infer_gui.py) (`infer_gui.InferenceGUI`, `infer_gui.parse_args`)  
- Example Train/Validation lists: [python/trainingData/manuldetect/Train.txt](python/trainingData/manuldetect/Train.txt), [python/trainingData/manuldetect/Validation.txt](python/trainingData/manuldetect/Validation.txt)

Feel free to run the trainer or GUI and inspect the code above for extensibility points (data discovery, YAML generation, tensorboard export).