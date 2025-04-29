@echo off
echo Cleaning build artifacts...
rd /s /q build 2>nul
rd /s /q dist 2>nul
for /d %%d in (*.egg-info) do rd /s /q "%%d" 2>nul
for /d %%d in (*.dist-info) do rd /s /q "%%d" 2>nul

echo Uninstalling existing package (if any)...
pip uninstall speech-to-text-cli -y > nul 2>&1

echo Building the project...
python -m build

echo Installing the new package...
cd dist
for %%f in (*.whl) do (
    pip install "%%f"
)
cd ..

echo Build and install process finished.