@echo off
cd /d "%~dp0"

where conda >nul 2>nul
if errorlevel 1 (
    echo conda was not found. Install Miniconda from https://docs.conda.io/en/latest/miniconda.html and try again.
    pause
    exit /b 1
)

call conda env list | findstr /b "mea_tool " >nul
if errorlevel 1 (
    echo Setting up environment ^(first run only, may take a few minutes^)...
    call conda env create -f environment.yml
)

call conda activate mea_tool
echo Starting MEA Spike Analyser -- this will open in your browser...
streamlit run app.py
