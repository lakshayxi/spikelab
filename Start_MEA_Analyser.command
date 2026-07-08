#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if ! command -v conda &> /dev/null; then
    echo "conda was not found. Install Miniconda from https://docs.conda.io/en/latest/miniconda.html and try again."
    read -p "Press Enter to exit..."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^mea_tool "; then
    echo "Setting up environment (first run only, may take a few minutes)..."
    conda env create -f environment.yml
fi

conda activate mea_tool
echo "Starting MEA Spike Analyser — this will open in your browser..."
streamlit run app.py
