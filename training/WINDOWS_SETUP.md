# Training the hand-drawn models on Windows (RTX 4060)

Follow these steps **in order**. The one mistake that breaks everything is
installing the CPU-only build of PyTorch — Step 3 prevents that, so don't skip it.

---

## 1. Install Python 3.11 or 3.12

Get it from <https://www.python.org/downloads/windows/>. On the first installer
screen, **tick "Add python.exe to PATH"**, then install.

Verify in a new `cmd` window:

```bat
python --version
```

If that says *"not recognized"*, use `py` everywhere instead of `python` (and
pass `set PY=py` before the training script — see Step 6).

## 2. Get the code

```bat
git clone https://github.com/<your-account>/Circuit-SPICE-generator.git
cd Circuit-SPICE-generator
git checkout training-scaffold
```

(Optional but recommended) make a virtual environment so nothing collides:

```bat
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install the CUDA build of PyTorch FIRST  ← the important step

Install PyTorch **from the CUDA index before anything else**. If you run
`pip install -r requirements.txt` first, pip grabs the **CPU-only** wheel from
PyPI and the GPU sits idle.

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> Not sure which CUDA version? The official picker gives you the exact command:
> <https://pytorch.org/get-started/locally/> → Stable · Windows · Pip · Python · CUDA 12.x.
> `cu124` works for any recent NVIDIA driver on a 40-series card.

## 4. Install the rest of the dependencies

Now pip sees PyTorch is already satisfied and won't touch it:

```bat
pip install -r requirements.txt
```

## 5. Verify the GPU is actually being used  ← must print `True`

```bat
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

- Prints `True NVIDIA GeForce RTX 4060` → you're good.
- Prints `False` → you have the CPU wheel. Redo Step 3 after removing it:
  ```bat
  pip uninstall -y torch torchvision
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```

## 6. (Optional) Add the real CGHD data for a better detector

Download Zenodo record **17469897** (`cghd-zenodo-16.zip`, ~4.9 GB) from
<https://zenodo.org/records/17469897>, and extract it so the `drafter_*` folders
end up under `datasets\cghd_raw\`, e.g. `datasets\cghd_raw\drafter_1\...`.
The training script auto-detects it. Skip this and it trains on synthetic data
only (fine for a first test, weaker detector).

## 7. Validate the whole pipeline (do this BEFORE the long run)

This runs the entire chain — generate data → train 2 epochs → run inference →
write a netlist — in about 3–5 minutes. It proves everything works on this PC.
The netlist won't be accurate yet; that's expected.

```bat
set SMOKE=1 & training\train_hand_drawn.bat
```

Look for **`PIPELINE OK`** and **`SMOKE TEST PASSED`** at the end.
(First run also downloads the EasyOCR text models, ~64 MB — needs internet once.)

## 8. Do the real training run

```bat
training\train_hand_drawn.bat
```

This trains both models for 100 epochs. Expect roughly **3–6 hours** total
(YOLO is the long part). The GPU will run near 100% and get hot — that's normal.
**Don't game or run other GPU apps while it trains** (they fight over VRAM).
Output weights land in `models\hand_drawn_*.pt`.

---

### Knobs (optional)

Set any of these before the command, e.g. `set EPOCHS=60 & training\train_hand_drawn.bat`:

| Variable | Default | Use when |
|---|---|---|
| `EPOCHS` | 100 | shorter/longer training |
| `BATCH_UNET` | 8 | drop to `4` if the U-Net hits "CUDA out of memory" |
| `BATCH_YOLO` | 8 | drop to `4` (or `IMG=640`) if YOLO hits OOM |
| `WORKERS` | 4 | set `0` if the U-Net data loader stalls at the start |
| `N_SYNTH` | 1500 | more/less synthetic data |
| `PY` | python | set `py` if `python` isn't on PATH |

### If something fails

The script stops at the first error and prints what to do. The usual culprit is
`torch.cuda.is_available()` being `False` — fix it with Step 5.
