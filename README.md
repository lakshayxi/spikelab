# 🧠 MEA Spike Analyser

Offline tool for Multi-Electrode Array spike detection, burst analysis, and amplitude quantification.  
Implements peer-reviewed burst detection methods ranked by Cotterill et al. (2016).

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

## Burst Detection Methods

Two peer-reviewed methods are available. You can run either or both simultaneously.

### Max Interval — Cotterill et al. (2016)  *recommended*
> Cotterill, E., et al. (2016). *A comparison of computational methods for detecting bursts in neuronal spike trains and their application to human stem cell-derived neuronal networks.* Journal of Neurophysiology, 116(2), 306–321.

Ranked **#1 of 8** burst detection methods across 11 desirable properties.  
Uses five independently tuneable parameters:

| Parameter | Default | Meaning |
|---|---|---|
| Max beginning ISI | 170 ms | Maximum ISI that opens a burst |
| Max end ISI | 300 ms | Maximum ISI that continues a burst |
| Min interburst interval | 200 ms | Minimum gap between separate bursts |
| Min burst duration | 10 ms | Shortest event counted as a burst |
| Min spikes per burst | 3 | Minimum event count |

### logISI Adaptive — Pasquale et al. (2010)
> Pasquale, V., et al. (2010). *A self-adapting approach for the detection of bursts and network bursts in neuronal cultures.* Journal of Computational Neuroscience, 29(1–2), 213–229.

Data-driven: derives the ISI threshold automatically from the logarithmic ISI histogram.

1. Bins ISIs in log₁₀ space (bin width = 0.1)
2. Smooths with a Gaussian kernel (σ = 1 bin)
3. Finds two principal peaks: intra-burst (< 100 ms) and inter-burst (≥ 100 ms)
4. Computes the **void parameter** between peaks:  
   `void = 1 − g(min) / √(g(peak₁) × g(peak₂))`
5. If void > threshold (default 0.7): sets ISIth at the histogram minimum
6. If ISIth > 100 ms: uses dual-threshold detection (100 ms for burst cores, ISIth to extend boundaries)
7. If no bimodal structure found: falls back to 100 ms Max Interval

### Both — compare methods
Runs both methods and computes the **normalised Hamming distance** (fraction of 50 ms bins where they disagree):

| Hamming distance | Interpretation |
|---|---|
| < 5% | Strong agreement ✅ — burst calls are robust |
| 5–10% | Moderate agreement ⚠️ — broadly consistent |
| > 10% | Poor agreement ❌ — review parameters or ISI structure |

---

## How to use

1. Upload your **raw data file** (two columns: time in seconds, voltage in µV)
2. Optionally upload your **NeuroExplorer export** to use its spike timestamps
3. Select a burst detection method in the sidebar
4. Adjust parameters — plots update instantly
5. Export any figure or the full data table as CSV
6. Click **Generate Methods Text** to get a ready-to-paste methods paragraph

---

## Parameters explained

| Parameter | What it does | Default |
|---|---|---|
| Threshold (× σ) | Spike detection: multiples of noise floor | 5.0 |
| Bandpass low/high | Frequency range kept after filtering | 300–3000 Hz |
| Burst detection method | MI, logISI, or both | Max Interval |
| Void parameter threshold | Minimum separation quality for logISI | 0.7 |
| Min spikes/burst | Minimum spikes to count as a burst | 3 |
| Pre/post window | Waveform cut around each spike | 1 ms / 2 ms |

---

## Output tabs

| Tab | Contents |
|---|---|
| Overview | Spike raster and instantaneous firing rate |
| ISI Analysis | Inter-spike interval histogram (linear + log), coloured by threshold |
| Amplitude | All waveforms overlaid, trough and P2P distributions, amplitude over time |
| Bursts | Per-burst duration/spike count, in-burst vs isolated spike amplitudes |
| logISI Histogram | Log-space ISI histogram with detected peaks, void parameter, and (in Both mode) a side-by-side comparison raster |
| Data Table | Every spike with timestamp, trough, P2P, SNR, and burst membership |

---

## File formats supported

**Raw data file** — tab- or comma-separated, two columns:
```
1.358   -1.192096
1.35804 -0.596048
```
Exported from Multi Channel Analyzer as "Raw Data Time Points".

**NeuroExplorer export** — standard ASCII export with Instantaneous Parameters section.

---

## References

- Cotterill, E., et al. (2016). A comparison of computational methods for detecting bursts in neuronal spike trains. *Journal of Neurophysiology*, 116(2), 306–321.
- Pasquale, V., et al. (2010). A self-adapting approach for the detection of bursts and network bursts in neuronal cultures. *Journal of Computational Neuroscience*, 29(1–2), 213–229.
- Quiroga, R. Q., Nadasdy, Z., & Ben-Shaul, Y. (2004). Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. *Neural Computation*, 16(8), 1661–1687.
- Obien, M. E. J., et al. (2015). Revealing neuronal function through microelectrode array recordings. *Frontiers in Neuroscience*, 8, 423.
