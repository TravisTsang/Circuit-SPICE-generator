# Circuits OCR

End-to-end schematic OCR pipeline for converting noisy, low-quality, or hand-drawn electrical schematic images into SPICE netlists.

The application is organized as a multi-stage machine-learning and graph-reconstruction pipeline:

1. A U-Net segments conductive traces from the noisy source image.
2. YOLOv8 detects schematic symbols such as resistors, capacitors, voltage sources, and ground.
3. EasyOCR reads nearby text labels and values.
4. The wire mask is skeletonized into one-pixel paths.
5. NetworkX stores the recovered component-to-net topology.
6. The exporter writes a SPICE `.cir` or `.net` file.

The current repository contains the inference application. You still need to train or provide the U-Net and YOLOv8 weights before running real images.

## Directory Structure

```text
Circuits OCR/
  README.md
  requirements.txt

  data/
    input/
      schematic.png                 # Raw images to convert
    output/
      schematic.cir                 # Generated SPICE files
      schematic_trace_mask.png      # Optional debug output
      schematic_skeleton.png        # Optional debug output
      schematic_topology.json       # Optional debug output
    samples/
      ...                           # Example images for experiments

  models/
    unet_trace_segmentation.pt      # Trained U-Net weights, created by you
    yolov8_components.pt            # Trained YOLOv8 weights, created by you

  statics_ocv/
    __init__.py                     # Lazy public package exports
    __main__.py                     # Allows python -m statics_ocv
    config.py                       # Thresholds, paths, SPICE prefixes
    segmentation.py                 # U-Net loading and wire-mask inference
    detection.py                    # YOLOv8 detection, EasyOCR, text association
    graph_builder.py                # Skeletonization and circuit topology recovery
    netlist_exporter.py             # SPICE line generation and file writing
    main.py                         # Command-line orchestration
```

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put trained weights here:

```text
models/unet_trace_segmentation.pt
models/yolov8_components.pt
```

Then run:

```powershell
python -m statics_ocv data\input\schematic.png --output data\output\schematic.cir --dump-intermediates
```

Use `--dump-intermediates` while developing. It saves the trace probability map, binary mask, skeleton image, net labels, topology JSON, and warnings so you can see where the pipeline is succeeding or failing.

## Web Application

The repo also includes a React + FastAPI web prototype:

```text
backend/
  server.py                  # FastAPI upload/inference API

frontend/
  package.json               # Vite React app
  src/App.jsx                # Paste/drop UI and result panels
  src/main.jsx
  src/index.css
  tailwind.config.js
```

### Web Model Weights

The web API supports two domains. Each domain has its own U-Net and YOLO weights:

```text
models/
  hand_drawn_unet_trace_segmentation.pt
  hand_drawn_yolov8_components.pt
  digital_unet_trace_segmentation.pt
  digital_yolov8_components.pt
```

You can override these paths with environment variables:

```powershell
$env:HAND_DRAWN_UNET_WEIGHTS="C:\path\to\hand_unet.pt"
$env:HAND_DRAWN_YOLO_WEIGHTS="C:\path\to\hand_yolo.pt"
$env:DIGITAL_UNET_WEIGHTS="C:\path\to\digital_unet.pt"
$env:DIGITAL_YOLO_WEIGHTS="C:\path\to\digital_yolo.pt"
$env:MODEL_DEVICE="cpu"
```

The backend caches one loaded model bundle per domain. Switching between Hand-Drawn Circuits and Digital Circuits does not reload weights on every request; the first request for a domain loads that domain's U-Net, YOLO, and OCR reader, and later requests reuse them.

### Run The Backend

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

### Run The Frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

The UI supports global image paste, drag-and-drop upload, a domain selector for Hand-Drawn Circuits vs. Digital Circuits, and a Process Circuit action. The frontend sends a `FormData` request containing:

```text
image=<uploaded image>
domain=hand-drawn | digital
```

The backend returns component boxes, OCR spans, recovered nets, warnings, timing, the active weight paths, and generated SPICE netlist text.

## How The Code Works

### 1. `main.py`

`main.py` is the entry point. It wires all stages together.

Important functions:

- `build_arg_parser()`: Defines the command-line interface, including input image, output path, config path, model overrides, device, and debug output.
- `main(argv=None)`: Parses CLI arguments, loads configuration, applies overrides, configures logging, and calls `run_pipeline()`.
- `run_pipeline(image_path, output_path=None, config=None, dump_intermediates=False)`: Runs the full application:
  - Loads the image path.
  - Runs `TraceSegmenter`.
  - Runs `ComponentDetector`.
  - Runs `CircuitGraphBuilder`.
  - Runs `NetlistExporter`.
  - Optionally writes intermediate debug files.
- `configure_logging(level)`: Sets a readable logging format.
- `_dump_intermediates(...)`: Writes debug artifacts such as skeleton PNGs, net label arrays, and topology JSON.

Data flow:

```text
image path
  -> TraceSegmenter.segment_image()
  -> ComponentDetector.detect()
  -> CircuitGraphBuilder.build()
  -> NetlistExporter.write()
  -> SPICE netlist
```

### 2. `config.py`

`config.py` centralizes settings so the processing thresholds and model paths are not scattered through the code.

Important classes:

- `ModelConfig`: Model paths and inference settings:
  - `unet_path`
  - `yolo_path`
  - `device`
  - `image_size`
  - `segmentation_threshold`
  - `yolo_confidence`
  - `yolo_iou`
  - EasyOCR language and GPU settings
- `GraphConfig`: Pixel-level topology settings:
  - minimum wire length
  - terminal search radius
  - endpoint gap warning distance
  - component bounding-box padding
  - text association distance
- `SpiceConfig`: Electrical export settings:
  - detector-class-to-SPICE-prefix mapping
  - expected terminal counts
  - default component values
  - default `.model` statements
- `AppConfig`: Top-level config object containing `models`, `graph`, and `spice`.

Important functions and methods:

- `AppConfig.from_yaml(path)`: Loads optional YAML overrides into the default config.
- `normalize_component_class(class_name)`: Normalizes detector labels such as `Voltage Source` into `voltage_source`.
- `spice_prefix_for(class_name)`: Converts a detector class into a SPICE prefix, such as `resistor` to `R`.
- `terminal_count_for_prefix(prefix)`: Returns how many terminals a component should have.
- `is_ground_class(class_name)`: Checks whether a detected class should be treated as SPICE node `0`.
- `_merge_dataclass(target, source)`: Internal recursive helper for YAML config merging.

### 3. `segmentation.py`

`segmentation.py` handles U-Net inference for conductive trace extraction.

Important classes:

- `DoubleConv`: Two convolution, batch-normalization, and ReLU blocks used inside the U-Net.
- `UNet`: A compact binary segmentation U-Net. It supports normal PyTorch state dictionaries. If your trained model uses a different architecture, export it as TorchScript and the loader will use that instead.
- `SegmentationResult`: Dataclass containing:
  - `probability_map`: floating-point trace probability image
  - `binary_mask`: thresholded clean wire mask
- `TraceSegmenter`: Main segmentation interface.

Important methods:

- `TraceSegmenter.__init__(config)`: Loads the trained U-Net from `models/unet_trace_segmentation.pt` or from the configured path.
- `segment_image(image_path)`: Reads an image with OpenCV and returns a `SegmentationResult`.
- `segment_array(image_bgr)`: Runs preprocessing, model inference, thresholding, and post-processing on an image array.
- `save_debug_outputs(result, output_dir, stem)`: Saves probability and mask images.
- `_load_model(path)`: Loads either TorchScript or a regular PyTorch checkpoint.
- `_preprocess(image_bgr)`: Converts BGR to RGB, resizes, normalizes, and converts to a tensor.
- `_postprocess_mask(mask)`: Morphologically closes small gaps and removes isolated noise.

How it works with the rest of the system:

- The output `binary_mask` is passed directly to `CircuitGraphBuilder.build()`.
- The mask should contain only conductive traces, not component bodies or text. Good segmentation quality is the most important part of accurate topology recovery.

### 4. `detection.py`

`detection.py` detects components and reads text.

Important data classes:

- `BoundingBox`: Stores `x1`, `y1`, `x2`, `y2` and provides width, height, center, area, expansion, and point-distance helpers.
- `OCRText`: Stores recognized text, its bounding box, and OCR confidence.
- `ComponentDetection`: Stores one detected schematic component:
  - detector class
  - SPICE prefix
  - bounding box
  - confidence
  - label such as `R1`
  - value such as `10k`
  - associated OCR text
  - terminal nets filled later by `graph_builder.py`
- `DetectionResult`: Contains all components and OCR spans.

Important methods:

- `ComponentDetector.__init__(config)`: Loads YOLOv8 and EasyOCR.
- `detect(image_path)`: Runs symbol detection and OCR, then associates text to components.
- `_load_yolo(path)`: Loads the trained YOLOv8 model.
- `_load_cv2()`: Imports OpenCV with a clear error message if missing.
- `_detect_components(image_bgr)`: Runs YOLOv8 and converts boxes into `ComponentDetection` objects.
- `_detect_text(image_bgr)`: Runs EasyOCR and converts OCR quadrilaterals into `OCRText`.
- `_associate_text(components, texts)`: Finds nearby OCR text for each component. It separates reference designators such as `R1` from values such as `10k`.
- `_fill_missing_component_names(components)`: Creates stable names like `R1`, `C1`, or `V1` when OCR did not read a label.
- `_association_score(component, text)`: Scores text by distance to the component.

Important helper functions:

- `clean_ocr_text(text)`: Removes whitespace and fixes common OCR artifacts.
- `parse_reference_designator(text)`: Parses labels like `R12`.
- `normalize_value_for_spice(text, prefix)`: Converts OCR values into SPICE-friendly strings, such as `5V` to `DC 5`.

How it works with the rest of the system:

- `ComponentDetection` objects are passed to `CircuitGraphBuilder.build()`.
- The graph builder fills `terminal_nets` and `terminal_points`.
- The netlist exporter uses the final component names, values, and terminal nets.

### 5. `graph_builder.py`

`graph_builder.py` turns the wire mask and detected components into an electrical graph.

Important data classes:

- `TerminalCandidate`: Represents a nearby wire candidate for a component terminal.
- `CircuitTopology`: Final recovered topology:
  - `graph`: NetworkX graph
  - `components`: components with terminal nets filled in
  - `net_names`: map from skeleton labels to SPICE node names
  - `warnings`: open-circuit and ambiguity warnings
  - `skeleton`: one-pixel wire mask
  - `net_label_image`: labeled wire-net image

Important methods:

- `CircuitGraphBuilder.__init__(config)`: Stores graph settings and warning state.
- `build(binary_mask, components)`: Main topology recovery method:
  - Skeletonizes the binary wire mask.
  - Labels connected wire-net components.
  - Warns about nearby disconnected endpoints.
  - Maps ground symbols to SPICE node `0`.
  - Attaches component terminals to nearby nets.
  - Returns `CircuitTopology`.
- `_skeletonize(binary_mask)`: Uses `skimage.morphology.skeletonize` to reduce traces to one-pixel paths.
- `_label_wire_nets(skeleton)`: Uses connected-component labeling to turn skeleton regions into candidate electrical nets.
- `_initial_net_names(net_points)`: Creates names such as `N001`, `N002`, and `N003`.
- `_map_ground_symbols(components, label_image, net_names)`: Converts nets touching ground symbols into node `0`.
- `_attach_component(...)`: Adds a component node to the NetworkX graph and connects each expected terminal to a net.
- `_terminal_candidates(component, label_image, expected_count)`: Finds wire pixels near the component bounding box.
- `_point_is_terminal_contact(x, y, bbox)`: Filters out pixels deep inside the component body and keeps pixels near symbol edges.
- `_best_point_for_anchors(points, anchors)`: Finds the closest wire pixel to expected terminal anchor points.
- `_assign_candidates_to_anchors(...)`: Assigns detected wire contacts to expected terminals.
- `_terminal_anchors(bbox, expected)`: Estimates where terminals should be based on component orientation and expected terminal count.
- `_warn_for_open_circuits(skeleton, label_image)`: Finds nearby endpoints on different nets and logs possible broken wires.
- `_skeleton_endpoints(skeleton, label_image)`: Finds skeleton pixels with one or zero neighbors.
- `_warn(message)`: Stores warnings without crashing the pipeline.

Important design choice:

The graph builder treats each connected skeleton component as a candidate electrical net. This is more robust than trying to infer every drawn wire segment first, because SPICE only needs to know which component terminals share the same node. If a wire is broken, the code logs a warning and creates an open node instead of failing.

### 6. `netlist_exporter.py`

`netlist_exporter.py` converts recovered topology into SPICE text.

Important data classes:

- `NetlistResult`: Contains generated netlist text and export warnings.

Important methods:

- `NetlistExporter.__init__(config)`: Stores SPICE settings.
- `export(topology, title="schematic_ocr")`: Builds the final text netlist.
- `write(topology, output_path, title="schematic_ocr")`: Calls `export()` and writes the `.cir` or `.net` file.
- `_component_line(component)`: Formats one SPICE element line based on prefix:
  - `R`, `C`, `L`: two-node passive components
  - `V`, `I`: voltage and current sources
  - `D`: diode with model
  - `Q`: BJT with model
  - `M`: MOSFET with model
  - `S`: voltage-controlled switch
  - `X`: subcircuit instance
- `_component_sort_key(component)`: Keeps output deterministic.

How it works with the rest of the system:

- It expects `ComponentDetection.terminal_nets` to be filled by `graph_builder.py`.
- If values are missing, it uses defaults from `SpiceConfig`.
- If a component has missing terminals, it fills them with `NC_*` open nodes and records warnings.

### 7. `__init__.py` and `__main__.py`

- `__init__.py`: Exposes the public package API. Heavy ML imports are lazy, so simple commands like `python -m statics_ocv --help` work even before installing every dependency.
- `__main__.py`: Calls `main()` so the package can be run with:

```powershell
python -m statics_ocv ...
```

## What You Need To Do Now

You currently do not have trained weights, so the next work is data and training.

1. Install the dependencies.
2. Collect schematic images that look like your target inputs:
   - scanned textbook schematics
   - phone photos
   - hand-drawn circuits
   - noisy or low-resolution diagrams
   - clean CAD schematics, if those are also in scope
3. Label two datasets:
   - A segmentation dataset for wires.
   - A detection dataset for components.
4. Train the U-Net wire segmenter.
5. Train the YOLOv8 component detector.
6. Put weights into `models/`.
7. Run the pipeline with `--dump-intermediates`.
8. Inspect failures, improve labels, retrain, and repeat.

The first useful milestone is not perfect SPICE output. The first milestone is:

```text
The U-Net mask contains only wires, and YOLO boxes tightly cover every component.
```

Once that is true, graph recovery and netlist export become much easier to debug.

## How To Train The U-Net Wire Segmenter

### Goal

Train a binary segmentation model that answers this question for every pixel:

```text
Is this pixel part of a conductive wire or connection trace?
```

The target mask should include:

- wires
- junction dots
- short terminal stubs that connect into components

The target mask should exclude:

- component bodies
- text labels and values
- paper background
- shadows
- grid lines
- noise

### Recommended Dataset Layout

Create a training dataset like this:

```text
datasets/
  trace_segmentation/
    images/
      train/
        img_0001.png
      val/
        img_0101.png
    masks/
      train/
        img_0001.png
      val/
        img_0101.png
```

Each mask should be a single-channel image:

- black background: `0`
- white wire pixels: `255`

Image and mask filenames should match.

### How To Label Masks

Use an annotation tool that supports semantic segmentation masks. Good options include CVAT, Label Studio, Roboflow, or a drawing workflow where you manually paint white wire pixels over a black background.

For hand-drawn schematics, label the intended wire center and width, not every noisy pen artifact. The model should learn the electrical intent, not the scanner noise.

### Training Strategy

Start small:

- 100 to 300 labeled images can prove the pipeline.
- 1,000+ labeled images is better for real robustness.
- Include heavy augmentation: blur, JPEG artifacts, shadows, rotation, perspective, contrast changes, broken strokes, and paper texture.

Use losses that handle thin lines well:

- Binary cross entropy plus Dice loss.
- Optionally add focal loss if the model misses thin wires.

Recommended metrics:

- Dice score for overall segmentation.
- Precision, to avoid hallucinated wires.
- Recall, to avoid broken traces.
- Skeleton connectivity checks, because a visually decent mask can still break the circuit graph.

### Saving Weights For This App

The loader in `segmentation.py` accepts either:

- TorchScript model saved as `models/unet_trace_segmentation.pt`
- PyTorch checkpoint containing a `state_dict`

The safest deployment option is TorchScript:

```python
import torch

model.eval()
example = torch.randn(1, 3, 768, 768)
scripted = torch.jit.trace(model, example)
scripted.save("models/unet_trace_segmentation.pt")
```

If you save a state dict instead, use the same `UNet` architecture from `segmentation.py` or include compatible `model_kwargs`:

```python
torch.save(
    {
        "model_kwargs": {
            "in_channels": 3,
            "out_channels": 1,
            "features": (32, 64, 128, 256),
        },
        "state_dict": model.state_dict(),
    },
    "models/unet_trace_segmentation.pt",
)
```

## How To Train The YOLOv8 Component Detector

### Goal

Train an object detector that finds each electrical symbol and assigns a class.

Recommended starting classes:

```text
resistor
capacitor
inductor
diode
voltage_source
current_source
ground
switch
bjt
mosfet
opamp
```

Keep class names aligned with `SpiceConfig.class_to_prefix` in `config.py`. If your YOLO class is named `battery`, the config maps it to `V`. If you add a new class, add it to the config.

### Recommended Dataset Layout

YOLO expects images and label text files:

```text
datasets/
  component_detection/
    images/
      train/
        img_0001.png
      val/
        img_0101.png
    labels/
      train/
        img_0001.txt
      val/
        img_0101.txt
    data.yaml
```

Each label file contains normalized boxes:

```text
class_id x_center y_center width height
```

Example `data.yaml`:

```yaml
path: datasets/component_detection
train: images/train
val: images/val
names:
  0: resistor
  1: capacitor
  2: inductor
  3: diode
  4: voltage_source
  5: current_source
  6: ground
  7: switch
  8: bjt
  9: mosfet
  10: opamp
```

### Training Command

After installing requirements, train with Ultralytics:

```powershell
yolo detect train model=yolov8s.pt data=datasets\component_detection\data.yaml imgsz=768 epochs=100 batch=8
```

For a faster experiment:

```powershell
yolo detect train model=yolov8n.pt data=datasets\component_detection\data.yaml imgsz=640 epochs=50 batch=8
```

After training, copy the best weights:

```powershell
copy runs\detect\train\weights\best.pt models\yolov8_components.pt
```

### Labeling Tips

- Draw boxes tightly around the symbol body, not the surrounding text.
- Do not include wire length in component boxes unless it is part of the symbol.
- Label ground symbols as `ground`.
- If a component is rotated, still use the same class.
- If handwritten symbols vary a lot, include several styles in training.

## What About OCR Training?

You do not need to train EasyOCR at first. Start with the pretrained OCR reader and improve the image pipeline around it.

If OCR performs poorly:

1. Crop text regions around components using YOLO boxes and nearby OCR spans.
2. Increase image resolution before OCR.
3. Add preprocessing for contrast and thresholding.
4. Add a correction layer for common electronics values:
   - `1OOk` should become `100k`
   - `O.1u` should become `0.1u`
   - `5v` should become `DC 5`
5. Only consider OCR fine-tuning after you have many labeled text examples.

## Recommended Training Order

Train in this order:

1. YOLOv8 component detector first.
   - You can visually inspect boxes quickly.
   - It gives you immediate feedback on class names and symbol coverage.
2. U-Net wire segmenter second.
   - It needs more careful pixel labels.
   - It has the biggest impact on topology correctness.
3. Run the full pipeline.
4. Fix the biggest failure source:
   - missed component boxes
   - bad OCR values
   - broken wire masks
   - incorrect terminal mapping

## Debugging The Pipeline

Run with:

```powershell
python -m statics_ocv data\input\schematic.png --output data\output\schematic.cir --dump-intermediates --log-level DEBUG
```

Then inspect:

- `*_trace_probability.png`: U-Net confidence.
- `*_trace_mask.png`: thresholded wire mask.
- `*_skeleton.png`: one-pixel wires used for graph recovery.
- `*_net_labels.npy`: labeled connected wire components.
- `*_topology.json`: components, terminal nets, OCR text, and warnings.
- final `.cir`: generated SPICE netlist.

Common warnings:

- `Possible broken wire`: Two skeleton endpoints are close but disconnected.
- `terminal has no nearby wire`: A component terminal could not be mapped to a net.
- `multiple terminals map to the same net`: The component may be shorted or the bounding box is too large.
- `extra nearby wire nets were ignored`: The component box may overlap crossing wires or nearby unrelated traces.

## Practical First Milestone

For your first working version, use a narrow scope:

- Only support resistors, capacitors, voltage sources, and ground.
- Use clean printed schematics first.
- Train YOLO on those classes.
- Train U-Net on clean and mildly noisy wire masks.
- Generate netlists for simple series and parallel circuits.

After that works, add hand-drawn images, more components, and heavier noise.

## Current Limitations

- The app needs trained model weights before real inference.
- Terminal order is inferred geometrically from bounding boxes, so complex symbols may need class-specific terminal logic later.
- EasyOCR is generic and may misread handwritten electrical values.
- Crossed wires without junction dots are hard to resolve from pixels alone.
- SPICE export uses default models for devices unless OCR or config provides specific model names.

These are normal constraints for this kind of system. The right way forward is iterative: train, run with debug outputs, inspect failures, improve labels and thresholds, retrain.
