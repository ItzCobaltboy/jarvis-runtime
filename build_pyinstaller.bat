@echo off
REM Alternate compile path: bundle jarvis-runtime into a standalone jarvis.exe
REM with PyInstaller instead of Nuitka. Kept alongside build.bat (Nuitka) as
REM a fallback/comparison build — Nuitka is the primary supported path.
REM
REM Run from an activated ryzen-ai-1.8.0 conda prompt, from the repo root:
REM     conda activate ryzen-ai-1.8.0
REM     build_pyinstaller.bat

setlocal

for /f "delims=" %%i in ('python -c "import onnxruntime, os; print(os.path.dirname(os.path.dirname(onnxruntime.__file__)))"') do set SITE_PACKAGES=%%i

if "%SITE_PACKAGES%"=="" (
    echo [build_pyinstaller.bat] ERROR: could not resolve onnxruntime's site-packages path.
    echo [build_pyinstaller.bat] Make sure the ryzen-ai-1.8.0 conda env is activated first.
    exit /b 1
)

set FLEXML_LIB=%SITE_PACKAGES%\flexml\flexml_extras\lib

if not exist "%FLEXML_LIB%\vaiml.dll" (
    echo [build_pyinstaller.bat] ERROR: vaiml.dll not found at %FLEXML_LIB%
    echo [build_pyinstaller.bat] The onnxruntime-vitisai / flexml packages must be installed
    echo [build_pyinstaller.bat] in this conda env for NPU support to be bundled.
    exit /b 1
)

echo [build_pyinstaller.bat] Bundling flexml NPU DLLs from: %FLEXML_LIB%

REM --onefile: single exe, self-extracts to a temp dir at runtime (sys._MEIPASS).
REM --windowed: no console window (equivalent to Nuitka's --windows-console-mode=disable).
REM --add-binary: flexml_lib DLLs land next to main.py inside the bundle -> main.py's
REM   _fixup_npu_dll_path() looks for them under sys._MEIPASS\flexml_lib.
REM --add-data: same idea for the Whisper package and config.yaml (loaded via
REM   Path(__file__)-relative lookups, which resolve fine under the extracted temp dir).
REM --collect-all for packages that use dynamic/plugin-style imports PyInstaller's
REM   static analysis can miss (mirrors Nuitka's --include-package flags).
REM NOTE: do NOT use python -O / --optimize here. Stripping docstrings breaks
REM   transformers' auto_docstring decorator (it introspects docstrings of its
REM   own output classes at class-definition time), causing a ValueError deep
REM   in modeling_bert.py the moment sentence-transformers loads a BERT model.
REM NOTE: sentencepiece is excluded outright. Neither tokenizer we actually use
REM   (BertTokenizerFast for the MiniLM intent model, WhisperTokenizer for
REM   Whisper) imports it — it's just a transitive install in the conda env
REM   from transformers' extras. But PyInstaller's binary-dependency analysis
REM   discovers it via the module graph and its isolated-subprocess import
REM   crashes with an access violation (a real PyInstaller/sentencepiece
REM   incompatibility, unrelated to PyInstaller version — 6.10.0 and 6.22.3
REM   both crash the same way). Since it's dead weight, exclude it rather than
REM   work around the crash.
python -m PyInstaller ^
  --name jarvis ^
  --onefile ^
  --windowed ^
  --noconfirm ^
  --clean ^
  --add-binary "%FLEXML_LIB%\*;flexml_lib" ^
  --add-data "Whisper;Whisper" ^
  --add-data "config.yaml;." ^
  --collect-all onnxruntime ^
  --collect-all winrt ^
  --collect-all pycaw ^
  --collect-submodules comtypes ^
  --exclude-module comtypes.test ^
  --collect-data openwakeword ^
  --collect-submodules openwakeword ^
  --exclude-module openwakeword.train ^
  --exclude-module onnxscript ^
  --exclude-module matplotlib ^
  --exclude-module tkinter ^
  --exclude-module torchaudio ^
  --exclude-module torio ^
  --exclude-module sentencepiece ^
  --distpath dist_pyinstaller ^
  --workpath build_pyinstaller ^
  main.py

if errorlevel 1 (
    echo [build_pyinstaller.bat] Build FAILED.
    exit /b 1
)

echo [build_pyinstaller.bat] Build OK: dist_pyinstaller\jarvis.exe
echo [build_pyinstaller.bat] Note: models/ (openWakeWord ONNX files) and
echo [build_pyinstaller.bat] Whisper/cache/ (NPU-compiled model cache) are NOT
echo [build_pyinstaller.bat] bundled — fetched/compiled on first run, same as
echo [build_pyinstaller.bat] running from source.

endlocal
