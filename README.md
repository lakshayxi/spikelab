# MEA Spike Analyser

Offline tool for Multi-Electrode Array spike detection, burst analysis, and amplitude quantification.
Implements peer-reviewed burst detection methods ranked by Cotterill et al. (2016).

For the complete reference manual — every parameter, algorithm, formula, and plot explained — see [DOCUMENTATION.md](DOCUMENTATION.md).

---

## Key Features

- Spike detection with an adjustable, noise-floor-relative threshold
- Two peer-reviewed burst detection methods (Max Interval, logISI Adaptive), runnable individually or side by side for comparison
- Full amplitude, waveform-shape, and burst-dynamics analysis suite (attenuation index, intra-burst decrement, burst-level correlations, and more)
- Multi-segment file upload with automatic time-based stitching, for recordings exported in sequential chunks
- Optional NeuroExplorer import
- Publication-ready exports: combined and per-panel figure downloads, full spike-level CSV export, and an auto-generated methods-section paragraph

---

## Setup — one click

**Prerequisite:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed (one-time install, if not already present).

1. Unzip `MEA_Spike_Analyser.zip`.
2. Double-click the launcher for your OS:
   - **Mac:** `Start_MEA_Analyser.command`
   - **Windows:** `Start_MEA_Analyser.bat`

The first run creates an isolated environment and installs dependencies automatically (a few minutes, one time only) — every run after that starts instantly. Your browser will open automatically at http://localhost:8501.

### Manual setup (for developers working on the source)

```
conda env create -f environment.yml
conda activate mea_tool
streamlit run app.py
```

Or with plain pip instead of conda:

```
pip install -r requirements.txt
streamlit run app.py
```

---

## Quick Start

1. Upload your **raw data file** (two columns: time in seconds, voltage in µV)
2. Optionally upload your **NeuroExplorer export** to use its spike timestamps
3. Select a burst detection method in the sidebar
4. Adjust parameters — plots update instantly
5. Export any figure or the full data table as CSV
6. Click **Generate Methods Text** to get a ready-to-paste methods paragraph

---

## Burst Detection Methods

Two peer-reviewed methods are available, runnable individually or together for comparison.

**Max Interval** — Cotterill et al. (2016), J Neurophysiol. Ranked #1 of 8 burst detection methods across 11 desirable properties. Uses five independently tuneable parameters (beginning/end ISI thresholds, interburst interval, minimum duration, minimum spike count).

**logISI Adaptive** — Pasquale et al. (2010), J Comput Neurosci. Data-driven: derives its ISI threshold automatically from the shape of the recording's own ISI distribution, rather than requiring a fixed value.

Running both methods together computes their agreement via normalised Hamming distance. See [DOCUMENTATION.md](DOCUMENTATION.md#8-burst-detection-methods--algorithm-details) for the full algorithm details, formulas, and agreement thresholds.

---

## Parameters, Output Tabs, and File Formats

Every sidebar parameter (with ranges and defaults), every output tab, and both supported input file formats (raw Multi Channel Analyzer export, with multi-segment auto-stitching, and NeuroExplorer export) are documented in full in [DOCUMENTATION.md](DOCUMENTATION.md).

---

## References

- Cotterill, E., et al. (2016). A comparison of computational methods for detecting bursts in neuronal spike trains. *Journal of Neurophysiology*, 116(2), 306–321.
- Pasquale, V., et al. (2010). A self-adapting approach for the detection of bursts and network bursts in neuronal cultures. *Journal of Computational Neuroscience*, 29(1–2), 213–229.
- Quiroga, R. Q., Nadasdy, Z., & Ben-Shaul, Y. (2004). Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. *Neural Computation*, 16(8), 1661–1687.
- Obien, M. E. J., et al. (2015). Revealing neuronal function through microelectrode array recordings. *Frontiers in Neuroscience*, 8, 423.
