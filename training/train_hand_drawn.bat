@echo off
rem ===========================================================================
rem One-shot HAND-DRAWN training for Windows. Builds data, trains the U-Net +
rem YOLO pair, then runs an end-to-end check (image -> .cir) to prove the whole
rem pipeline is wired up.
rem
rem Produces:
rem   models\hand_drawn_unet_trace_segmentation.pt   (+ _torchscript.pt)
rem   models\hand_drawn_yolov8_components.pt
rem
rem TWO MODES:
rem   Validate first (fast, ~3-5 min):   set SMOKE=1 & training\train_hand_drawn.bat
rem   Then the real run:                 training\train_hand_drawn.bat
rem
rem The SMOKE run uses tiny data + 2 epochs: the netlist won't be accurate, it
rem just proves data -> train -> weights -> inference all work on this machine.
rem
rem CGHD (real boxes) is optional but recommended: put Zenodo record 17469897
rem into datasets\cghd_raw\ first; this script auto-detects it (real run only).
rem
rem If 'python' is not on PATH, use the launcher:  set PY=py & training\train_hand_drawn.bat
rem ===========================================================================
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0.."

if not defined PY set "PY=python"
if not defined SMOKE set "SMOKE=0"
if not defined YOLO_BASE set "YOLO_BASE=yolov8s.pt"
if not defined WORKERS set "WORKERS=4"
if not defined IMG set "IMG=768"
if not defined BATCH_UNET set "BATCH_UNET=8"
if not defined BATCH_YOLO set "BATCH_YOLO=8"

if "%SMOKE%"=="1" (
  if not defined N_SYNTH set "N_SYNTH=60"
  if not defined EPOCHS set "EPOCHS=2"
  echo ############################################################
  echo #  SMOKE TEST - validates the pipeline runs end to end.    #
  echo #  The netlist will NOT be accurate yet ^(only 2 epochs^).   #
  echo ############################################################
) else (
  if not defined N_SYNTH set "N_SYNTH=1500"
  if not defined EPOCHS set "EPOCHS=100"
)

echo === Python ===
%PY% --version || goto :nopy
%PY% -c "import torch; print('torch', torch.__version__, '| built for CUDA:', torch.version.cuda)" || goto :notorch

%PY% -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 goto :nocuda
%PY% -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

rem TLS fix so Ultralytics can auto-download yolov8 base weights (best-effort).
for /f "delims=" %%i in ('%PY% -m certifi 2^>nul') do set "SSL_CERT_FILE=%%i"
if defined SSL_CERT_FILE set "REQUESTS_CA_BUNDLE=%SSL_CERT_FILE%"

echo.
echo === [1/5] Generating %N_SYNTH% synthetic hand_drawn images ===
%PY% -m training.synthetic.generate --n %N_SYNTH% --domain hand_drawn || goto :err

echo.
if "%SMOKE%"=="1" (
  echo === [2/5] ^(smoke^) skipping CGHD ingest to stay fast ===
  goto :train
)
dir /b /s "datasets\cghd_raw\*.xml" >nul 2>&1
if errorlevel 1 goto :nocghd
echo === [2/5] CGHD detected - ingesting real boxes + weak wire masks ===
%PY% -m training.ingest.cghd_to_yolo --src datasets\cghd_raw || goto :err
%PY% -m training.ingest.cghd_wire_masks --src datasets\cghd_raw || goto :err
goto :train
:nocghd
echo === [2/5] No CGHD under datasets\cghd_raw\ - training detector on synthetic only ===
echo     Download CGHD ^(Zenodo 17469897^) into datasets\cghd_raw\ for a usable detector.

:train
echo.
echo === [3/5] Training U-Net wire segmenter ^(AMP/bf16, batch %BATCH_UNET%, %EPOCHS% epochs^) ===
%PY% -m training.unet.train_unet --data datasets\trace_segmentation --epochs %EPOCHS% --batch %BATCH_UNET% --imgsz %IMG% --device cuda --amp --amp-dtype bf16 --workers %WORKERS% --out models\hand_drawn_unet_trace_segmentation.pt || goto :err

echo.
echo === [4/5] Training YOLOv8 component detector ^(batch %BATCH_YOLO%, %EPOCHS% epochs^) ===
%PY% -m training.yolo.train_yolo --data training\yolo\data.yaml --model %YOLO_BASE% --imgsz %IMG% --epochs %EPOCHS% --batch %BATCH_YOLO% --device 0 --name hand_drawn_components --out models\hand_drawn_yolov8_components.pt || goto :err

echo.
echo === [5/5] End-to-end pipeline check ^(image -^> .cir using the new weights^) ===
set "SAMPLE="
for /f "delims=" %%f in ('dir /b /s "datasets\component_detection\images\train\synth_hand_drawn_*.png" 2^>nul') do if not defined SAMPLE set "SAMPLE=%%f"
if not defined SAMPLE for /f "delims=" %%f in ('dir /b /s "datasets\component_detection\images\*.png" 2^>nul') do if not defined SAMPLE set "SAMPLE=%%f"
if not defined SAMPLE (
  echo     ^(no sample image found to validate inference - skipping^)
  goto :done
)
echo     input: !SAMPLE!
echo     ^(first inference downloads the EasyOCR models, ~64MB - needs internet once^)
%PY% -m statics_ocv "!SAMPLE!" --output data\output\_pipeline_check.cir --unet-model models\hand_drawn_unet_trace_segmentation.pt --yolo-model models\hand_drawn_yolov8_components.pt --dump-intermediates || goto :checkfail
for %%A in ("data\output\_pipeline_check.cir") do if %%~zA GTR 0 ( goto :checkok ) else ( goto :checkfail )

:checkok
echo.
echo     ===================  PIPELINE OK  ===================
echo     Netlist written: data\output\_pipeline_check.cir

:done
echo.
if "%SMOKE%"=="1" (
  echo === SMOKE TEST PASSED. The pipeline runs end to end. ===
  echo     Now do the real run:   training\train_hand_drawn.bat
) else (
  echo === DONE. Hand-drawn weights: ===
  dir models\hand_drawn_*.pt
  echo.
  echo Wire them into the backend ^(PowerShell^):
  echo   $env:HAND_DRAWN_UNET_WEIGHTS = "%CD%\models\hand_drawn_unet_trace_segmentation.pt"
  echo   $env:HAND_DRAWN_YOLO_WEIGHTS = "%CD%\models\hand_drawn_yolov8_components.pt"
)
goto :eof

:checkfail
echo PIPELINE CHECK FAILED: inference did not produce a netlist (see output above). >&2
exit /b 1
:nopy
echo ERROR: '%PY%' not found. Install Python 3, or run:  set PY=py ^& training\train_hand_drawn.bat
exit /b 1
:notorch
echo ERROR: PyTorch not importable. Run:  %PY% -m pip install -r requirements.txt
exit /b 1
:nocuda
echo ERROR: torch.cuda.is_available() is False - you have the CPU-only PyTorch wheel.
echo        Fix it with:
echo          %PY% -m pip uninstall -y torch torchvision
echo          %PY% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
exit /b 1
:err
echo ERROR: a step failed (see output above).
exit /b 1
