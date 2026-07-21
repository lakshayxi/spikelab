@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo Checking Miniconda...

set "CONDA="

where conda >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%i in ('where conda') do (
        if not defined CONDA set "CONDA=%%i"
    )
)

if not defined CONDA (
    for %%P in (
        "%USERPROFILE%\miniconda3\condabin\conda.bat"
        "%USERPROFILE%\anaconda3\condabin\conda.bat"
        "%ProgramData%\miniconda3\condabin\conda.bat"
        "%ProgramData%\anaconda3\condabin\conda.bat"
    ) do (
        if not defined CONDA if exist %%P set "CONDA=%%~P"
    )
)

if not defined CONDA (
    echo ERROR: Miniconda/Anaconda was not found on PATH or in the usual install locations.
    echo Install Miniconda from https://docs.conda.io/en/latest/miniconda.html and try again.
    pause
    exit /b 1
)

echo Miniconda found: %CONDA%

echo Checking environment: mea_tool

call "%CONDA%" env list | findstr /b "mea_tool " >nul

if errorlevel 1 (
    echo.
    echo Environment not found.
    echo Creating mea_tool environment...
    echo This may take several minutes.
    echo.

    call "%CONDA%" env create -f environment.yml

    if errorlevel 1 (
        echo ERROR: Failed to create Conda environment.
        pause
        exit /b 1
    )
)

echo Activating mea_tool environment...

call "%CONDA%" activate mea_tool

if errorlevel 1 (
    echo ERROR: Failed to activate mea_tool environment.
    pause
    exit /b 1
)

echo.
echo Starting MEA Spike Analyser...
echo Your browser should open automatically.
echo.

python -m streamlit run app.py

pause
