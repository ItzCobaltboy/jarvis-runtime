@echo off
REM Compile jarvis-runtime to a standalone jarvis.exe with Nuitka.
REM
REM Run from an activated ryzen-ai-1.8.0 conda prompt, from the repo root:
REM     conda activate ryzen-ai-1.8.0
REM     build.bat
REM
REM Rebuild any time source changes — this is the only supported build path.
REM Do NOT delete main.py/listener.py/etc after compiling; build.bat is meant
REM to be re-run, not a one-shot throwaway.

setlocal

REM --- Locate the active conda env's site-packages, so we can find and
REM     bundle the flexml NPU DLLs (vaiml.dll etc), which onnxruntime loads
REM     dynamically at runtime and Nuitka's import scanner cannot see. ---
for /f "delims=" %%i in ('python -c "import onnxruntime, os; print(os.path.dirname(os.path.dirname(onnxruntime.__file__)))"') do set SITE_PACKAGES=%%i

if "%SITE_PACKAGES%"=="" (
    echo [build.bat] ERROR: could not resolve onnxruntime's site-packages path.
    echo [build.bat] Make sure the ryzen-ai-1.8.0 conda env is activated first.
    exit /b 1
)

set FLEXML_LIB=%SITE_PACKAGES%\flexml\flexml_extras\lib

if not exist "%FLEXML_LIB%\vaiml.dll" (
    echo [build.bat] ERROR: vaiml.dll not found at %FLEXML_LIB%
    echo [build.bat] The onnxruntime-vitisai / flexml packages must be installed
    echo [build.bat] in this conda env for NPU support to be bundled.
    exit /b 1
)

echo [build.bat] Bundling flexml NPU DLLs from: %FLEXML_LIB%

python -m nuitka ^
  --onefile ^
  --assume-yes-for-downloads ^
  --windows-console-mode=disable ^
  --enable-plugin=pyqt6 ^
  --nofollow-import-to=datasets ^
  --nofollow-import-to=onnxscript ^
  --module-parameter=torch-disable-jit=yes ^
  --include-package=onnxruntime ^
  --include-package=winrt ^
  --include-package=pycaw ^
  --include-package=comtypes ^
  --include-data-dir="%FLEXML_LIB%"=flexml_lib ^
  --include-data-dir=Whisper=Whisper ^
  --include-data-file=config.yaml=config.yaml ^
  --output-filename=jarvis.exe ^
  --output-dir=dist ^
  main.py

if errorlevel 1 (
    echo [build.bat] Build FAILED.
    exit /b 1
)

echo [build.bat] Build OK: dist\jarvis.exe
echo [build.bat] Note: models/ (openWakeWord ONNX files, downloaded on first
echo [build.bat] run) and Whisper/cache/ (NPU-compiled model cache) are NOT
echo [build.bat] bundled — they are fetched/compiled on the exe's first run,
echo [build.bat] same as when running from source.

endlocal
