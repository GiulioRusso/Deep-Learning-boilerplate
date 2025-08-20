# 🧠 Deep Learning Boilerplate Project

This repository is a **minimal, customizable boilerplate** for deep learning experiments built on **PyTorch Lightning**. It provides a clean entry point (`main.py`), a ready-to-edit LightningModule (`net/model/LitModel.py`), and utilities to organize **training**, **resume**, and **testing** runs into structured experiment folders.

> This README is tailored to the code in this repo (unpacked and inspected):
>
> * Experiment root folder: **`experiments/`**
> * Logger: **`TensorBoardLogger(save_dir="experiments", name=exp_ID, version=...)`**
> * Human‑readable ops log: **`log.txt`** inside each version folder (via `net.utility.logger.log_operation`).

---

## 📦 Project Structure

```
.
├── main.py                          # Entry point: train / resume / test
├── requirements.txt                 # Dependencies
├── experiments/                     # (Generated) Experiment runs live here
├── configs/
│   ├── parameters.yaml              # CLI params definitions & defaults
│   ├── paths.yaml                   # Dataset and (optional) experiment paths
│   └── statistics.yaml              # Example dataset stats (if needed)
├── net/
│   ├── __init__.py
│   ├── dataset/
│   │   ├── classes/MyDataset.py     # Example dataset class (customize)
│   │   ├── dataset_split.py         # Split train/val/test
│   │   └── dataset_transforms.py    # Transform pipeline
│   ├── initialization/
│   │   ├── experiment_ID.py         # Builds `exp_ID` string
│   │   ├── parameters_ID.py         # Pieces used by `exp_ID`
│   │   └── utility/get_yaml.py      # YAML loader
│   ├── model/
│   │   ├── LitModel.py              # LightningModule (edit your model here)
│   │   └── subnets/SubNet.py        # Minimal example submodule
│   ├── loss/                        # Loss factory
│   ├── optimizer/                   # Optimizer factory
│   ├── scheduler/                   # Scheduler factory
│   ├── parameters/parameters.py     # CLI parser (subcommands: train/resume/test)
│   ├── reproducibility/reproducibility.py  # Seeding & determinism
│   └── utility/
│       ├── get_latest_version.py    # Picks latest `version_x`
│       └── logger.py                # Appends to `log.txt`
├── scripts/                         # Aux scripts (examples)
├── data/                            # (Your datasets live outside or here)
└── notebooks/                       # (Optional) analysis notebooks
```

---

## 🧭 How experiments are named and organized

### `exp_ID`

`exp_ID` is generated in `main.py` via `experiment_ID(parser)`, which internally uses pieces defined in `net/initialization/parameters_ID.py`.

The resulting string concatenates key settings (dataset, net, transform, epochs, bs, lr, optimizer, scheduler, loss, …) with `|` separators.
**Example** (illustrative):

```
my_project|dataset=my_dataset|net=MyNet|transform=augV1|ep=100|bs=64|scheduler=cos|lr=1e-3|optimizer=adam|loss=bce
```

### Folder layout produced by the code

All runs are grouped under **`experiments/<exp_ID>/`** with Lightning‑managed **`version_*`** folders.

* **Fresh train** creates `version_0`, then `version_1`, … for subsequent new runs.
* **Resume** continues **inside the same version folder** (default `version=latest`, selectable via `--version`).
* **Test** also targets a specific version (default `latest`).

```
experiments/
└── <exp_ID>/
    └── version_0/
        ├── checkpoints/
        │   ├── best.ckpt            # Best by monitored metric (from LitModel callbacks)
        │   └── last.ckpt            # Last epoch/step (from LitModel callbacks)
        ├── events.out.tfevents.*    # TensorBoard logs (metrics, curves)
        └── log.txt                  # Human‑readable audit trail of runs
```

> **Where do these files come from?**
>
> * **Checkpoints** are controlled by `LitModel._get_checkpoint_callbacks()` (best + last).
> * **TensorBoard logs** are produced by `TensorBoardLogger` attached to the Trainer.
> * **`log.txt`** lines are appended by `net.utility.logger.log_operation(logger.log_dir, ...)` that `main.py` calls for each mode.

---

## ▶️ Typical workflows & what the code generates

> Commands use **subcommands** (`train`, `resume`, `test`) from `net/parameters/parameters.py`.

### 1) Train from scratch

```bash
python main.py train \
  --epochs 100 \
  --batch_size_train 64 \
  --batch_size_val 64 \
  --lr 1e-3 \
  --dataset my_dataset \
  --network_name MyNet
```

**What happens**

* `exp_ID` is built from your CLI/defaults.
* Logger writes to: `experiments/<exp_ID>/version_0/` (or the next free `version_x`).
* `log.txt` gets a line like:

  ```
  [YYYY-MM-DD HH:MM:SS] MODE=TRAIN | epochs=100 batch_size_train=64 lr=0.001 ...
  ```
* Checkpoints saved to `checkpoints/{best.ckpt,last.ckpt}`.
* View metrics in TensorBoard:

  ```bash
  tensorboard --logdir experiments/<exp_ID>
  ```

### 2) Resume training

```bash
# resumes the latest version of this experiment
python main.py resume --epochs 50 --version latest

# or target a specific version explicitly
python main.py resume --epochs 50 --version version_0
```

**What happens**

* `main.py` picks the version: `latest` → `net/utility/get_latest_version.py` or your explicit `--version`.
* Loads **`last.ckpt`** from `experiments/<exp_ID>/<version>/checkpoints/`.
* Continues training **in the same folder**, extending TensorBoard curves.
* Appends a new line to `log.txt`:

  ```
  [YYYY-MM-DD HH:MM:SS] MODE=RESUME | epochs=50 ckpt=.../checkpoints/last.ckpt ...
  ```

### 3) Test

```bash
# tests the latest version by default
python main.py test --version latest

# or a specific version
python main.py test --version version_0
```

**What happens**

* Loads **`best.ckpt`** from the selected version.
* Runs `test_step` and logs metrics to TensorBoard.
* Appends a line to `log.txt`:

  ```
  [YYYY-MM-DD HH:MM:SS] MODE=TEST | ckpt=.../checkpoints/best.ckpt ...
  ```
* (If implemented in your `LitModel`) writes CSV metrics next to checkpoints.

---

## ⚙️ Configuration

* **CLI defaults** are defined in `configs/parameters.yaml` and loaded by `net/parameters/parameters.py`.
* **Paths** are defined in `configs/paths.yaml` and read with `get_yaml(...)`.
* **Seeding**: `net/reproducibility/reproducibility.py` sets seeds and deterministic flags via `seed_everything`.

> You can override any default from the command line. The subparsers `train | resume | test` expose only the relevant flags per mode.

---

## ✍️ Customizing the template

* Implement your architecture and metrics in **`net/model/LitModel.py`**.
* Replace `net/dataset/classes/MyDataset.py` and the transform/split utilities with your dataset logic.
* Extend `parameters.yaml` with project‑specific flags (they will automatically appear in the CLI).
* Customize `experiment_ID.py` to include/exclude fields in the folder name.

---

## 🧪 Tips

* Keep `experiments/` under version control ignore (.gitignore) to avoid pushing large logs.
* Use TensorBoard to compare multiple versions under the same `exp_ID`.
* If you want a visible “resume marker” in TensorBoard, add a small callback that logs a scalar/text at `on_fit_start` when `ckpt_path` is set (optional enhancement).

---

## 🚨 Disclaimer
This project is not a fully operational project out-of-the-box. Instead, it serves as a structured starting point that needs to be customized based on your dataset, task requirements (such as classification, detection, or segmentation), and model configurations.
For any questions or support, feel free to reach out. Contributions that improve usability are welcome! You are encouraged to fork the repository and submit your modifications.

