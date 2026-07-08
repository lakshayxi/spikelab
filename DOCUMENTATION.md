# MEA Spike Analyser — Documentation

A complete reference manual for the MEA Spike Analyser: what it does, how to use every control, how each plot and metric should be interpreted, the exact algorithms and formulas behind the analysis, and how to export results for a publication methods section.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation & Setup](#2-installation--setup)
3. [Quick Start Workflow](#3-quick-start-workflow)
4. [Input File Formats](#4-input-file-formats)
5. [The Analysis Pipeline](#5-the-analysis-pipeline)
6. [Sidebar Parameter Reference](#6-sidebar-parameter-reference)
7. [Spike Detection — Algorithm Details](#7-spike-detection--algorithm-details)
8. [Burst Detection Methods — Algorithm Details](#8-burst-detection-methods--algorithm-details)
9. [Tab-by-Tab Reference](#9-tab-by-tab-reference)
10. [Metrics & Formulas Glossary](#10-metrics--formulas-glossary)
11. [Exporting for Publication](#11-exporting-for-publication)
12. [Known Limitations](#12-known-limitations)
13. [Glossary of Terms](#13-glossary-of-terms)
14. [References](#14-references)

---

## 1. Overview

The MEA Spike Analyser is an offline tool for analysing extracellular voltage recordings from Multi-Electrode Array (MEA) experiments. Given a raw voltage trace from a single recording channel, it detects spikes, identifies bursts using two independent peer-reviewed algorithms, quantifies spike amplitude and waveform shape, characterizes burst-level amplitude dynamics, and produces publication-ready figures, a spike-level data table, and a draft methods paragraph.

**Scope note:** the tool currently analyses **one channel at a time** — it does not yet support multi-electrode spatial analysis (e.g. electrode-array heatmaps or per-channel comparisons). Each analysis run corresponds to a single voltage trace, optionally reconstructed from several sequential export segments (see [Section 4](#4-input-file-formats)).

---

## 2. Installation & Setup

### One-click setup (recommended for lab use)

**Prerequisite:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed (one-time install).

1. Unzip the distributed package (`MEA_Spike_Analyser.zip`).
2. Double-click the launcher for your operating system: `Start_MEA_Analyser.command` (Mac) or `Start_MEA_Analyser.bat` (Windows).

The first launch automatically creates an isolated environment and installs all dependencies (a few minutes, one time only). Every subsequent launch starts immediately. The app opens automatically in your browser at `http://localhost:8501`.

### Manual setup (for developers working on the source)

Requires Python 3.9 or newer.

```bash
conda env create -f environment.yml
conda activate mea_tool
streamlit run app.py
```

Or with plain pip instead of conda:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dependencies: `streamlit`, `numpy`, `pandas`, `matplotlib`, `scipy`.

---

## 3. Quick Start Workflow

0. **Launch the app.** Double-click the launcher for your OS (see [Section 2](#2-installation--setup)) — no manual environment setup needed.
1. **Upload your raw data file(s).** A single file, or multiple sequential segment files from the same recording (they will be auto-ordered and stitched — see [Section 4](#4-input-file-formats)).
2. **(Optional) Upload a NeuroExplorer export.** If provided, its spike timestamps are used in place of the raw-signal spike detector for all downstream analysis.
3. **Choose a burst detection method** in the sidebar: Max Interval, logISI Adaptive, or Both (to compare).
4. **Adjust parameters** as needed — every plot and table updates immediately.
5. **Review the tabs** — spike raster, ISI histograms, amplitude/waveform quantification, burst statistics and dynamics, and the full spike-level data table.
6. **Export your results** — download any figure (combined or panel-by-panel), download the full spike table as CSV, and generate a ready-to-paste methods paragraph.

---

## 4. Input File Formats

### Raw voltage trace file

A two-column file — **time (seconds)** and **voltage (µV)** — tab- or comma-separated, as exported from Multi Channel Analyzer as **"Raw Data Time Points."** Typical export files include a short header (software name/version, channel label, column units) before the data rows; this header is detected and skipped automatically, no manual editing is required. Example data rows:

```
1.358     -1.192096
1.35804   -0.596048
1.35808   2.682216
```

The file uploader accepts `.txt` and `.csv` — the delimiter (tab or comma) is auto-detected from the file contents, so either export style works without conversion.

### Multi-segment upload & auto-stitching

Multi Channel Analyzer typically caps a single raw-trace export at around 10–12 seconds, so a full recording is often exported as several sequential files (e.g. `L5_1.txt`, `L5_2.txt`, `L5_3.txt`, ...). **Upload all segment files together** in the raw-trace uploader (it accepts multiple files at once) and the tool will:

1. **Reorder them by their own first timestamp** — regardless of the order you selected/uploaded them in, or how they're named. This means files can be uploaded in any order and will still be assembled correctly.
2. **Trim overlapping time regions.** Real export segments frequently overlap slightly at their boundaries (the end of one file's timestamps and the start of the next file's timestamps cover the same moment in time). The tool detects this automatically and keeps the earlier segment's data for any overlapping span, only appending the newer segment's genuinely new (later) samples — this avoids duplicated or aliased samples in the combined trace.
3. **Show a stitch summary.** After upload, an expandable panel reports the order the segments were assembled in, each segment's start/end time, how many samples were trimmed for overlap at each boundary, and the final combined recording duration — so you can visually verify the reconstruction is correct before trusting the analysis.

A single uploaded file skips all of this and is used directly, with identical behavior to previous versions of the tool.

### NeuroExplorer export (optional)

A standard NeuroExplorer ASCII export containing an **"Instantaneous Parameters"** section (spike time, ISI, and instantaneous firing rate columns). If uploaded, its spike timestamps are used for *all* downstream burst/amplitude/waveform analysis instead of the raw-signal spike detector — useful when you've already curated spike times in NeuroExplorer and want this tool's burst/amplitude analyses applied to that curated spike list. Note: the raw-signal spike detector still runs in the background regardless (to establish the noise floor and SNR reference), but its detected spike *times* are only used when no NeuroExplorer file is supplied.

Files missing the "Instantaneous Parameters" section will parse to an empty spike list rather than raising an error — if you upload a NeuroExplorer file and see no spikes, check that this section is present in the export.

---

## 5. The Analysis Pipeline

Every uploaded recording passes through the same sequence of steps:

1. **Parse** — raw time/voltage columns are read from the uploaded file(s), and (if applicable) multiple segments are stitched into one continuous trace.
2. **Determine sampling rate** — inferred automatically from the median spacing between consecutive timestamps. There is no manual override, so irregular or corrupted time columns will produce an incorrect sampling rate.
3. **Bandpass filter** — a 4th-order Butterworth filter (default 300–3000 Hz) removes slow drift and high-frequency noise, isolating the spike-frequency band.
4. **Spike detection** — negative threshold crossings on the filtered signal are detected, using a noise-floor-relative threshold and a refractory period to prevent double-counting (see [Section 7](#7-spike-detection--algorithm-details)).
5. **Burst detection** — the detected (or NeuroExplorer-supplied) spike train is analysed for bursts using the selected method(s) (see [Section 8](#8-burst-detection-methods--algorithm-details)).
6. **Waveform extraction** — a short window of the filtered signal around each spike is extracted, from which trough, peak, peak-to-peak amplitude, and spike width are computed.
7. **Derived metrics** — SNR, preceding/following ISI, burst-wise amplitude statistics, intra-burst amplitude decrement, and burst-level correlations are computed from the extracted waveforms and detected bursts.
8. **Presentation & export** — all of the above are rendered across the tool's tabs as figures and tables, exportable as PNG (combined figure or individual panels) and CSV, alongside an auto-generated methods paragraph.

---

## 6. Sidebar Parameter Reference

### Spike Detection

| Parameter | Range | Default | Step | What it controls |
|---|---|---|---|---|
| Threshold multiplier (× σ) | 3.0 – 10.0 | 5.0 | 0.5 | Spike detection threshold, as a multiple of the estimated noise floor. Standard is 5×; higher values are more conservative (fewer, higher-confidence spikes). |
| Bandpass low (Hz) | 100 – 1000 | 300 | 50 | Low cutoff of the bandpass filter applied before spike detection. |
| Bandpass high (Hz) | 1000 – 6000 | 3000 | 100 | High cutoff of the bandpass filter. |

### Burst Detection Method

A single selector chooses which method(s) run: **Max Interval (Cotterill et al. 2016)**, **logISI Adaptive (Pasquale et al. 2010)**, or **Both — compare methods**. Selecting a method reveals only its relevant parameters below.

**Max Interval parameters** (shown for Max Interval or Both):

| Parameter | Range | Default | Step | Meaning |
|---|---|---|---|---|
| Max beginning ISI (ms) | 50 – 500 | 170 | 10 | Longest inter-spike interval that can *open* a new burst. |
| Max end ISI (ms) | 100 – 1000 | 300 | 10 | Longest inter-spike interval that can *continue* an already-open burst. |
| Min interburst interval (ms) | 50 – 1000 | 200 | 10 | Minimum gap required between two candidate bursts for them to be counted as separate events (shorter gaps cause them to merge). |
| Min burst duration (ms) | 5 – 200 | 10 | 5 | Shortest event duration that still counts as a burst. |

**logISI parameters** (shown for logISI Adaptive or Both):

| Parameter | Range | Default | Step | Meaning |
|---|---|---|---|---|
| Void parameter threshold | 0.0 – 1.0 | 0.7 | 0.05 | Minimum separation quality required between the intra-burst and inter-burst ISI peaks before the data-driven threshold is trusted (see [Section 8](#8-burst-detection-methods--algorithm-details)). |

**Shared:**

| Parameter | Range | Default | Step | Meaning |
|---|---|---|---|---|
| Min spikes per burst | 2 – 10 | 3 | 1 | Minimum number of spikes required for an event to be counted as a burst. Applies to both Max Interval and logISI. |

### Waveform Extraction

| Parameter | Range | Default | Step | Meaning |
|---|---|---|---|---|
| Pre-spike window (ms) | 0.5 – 3.0 | 1.0 | 0.25 | How much signal before the threshold crossing is included in each extracted spike waveform. |
| Post-spike window (ms) | 1.0 – 5.0 | 2.0 | 0.25 | How much signal after the threshold crossing is included. |

---

## 7. Spike Detection — Algorithm Details

1. **Noise floor estimation.** The noise floor is estimated from the bandpass-filtered signal using the median absolute deviation (MAD) convention popularized by Quiroga et al. (2004):

   ```
   noise_floor = median(|filtered_signal|) / 0.6745
   ```

   The 0.6745 constant converts the median absolute deviation into an estimate equivalent to one standard deviation for Gaussian-distributed noise, making the estimate robust to the large-amplitude spikes themselves (which would otherwise inflate a naive standard-deviation estimate).

2. **Threshold.** `threshold = -(threshold_multiplier × noise_floor)`. The negative sign reflects that this tool detects **negative-going** threshold crossings only (the typical polarity of extracellular action potentials).

3. **Refractory period.** After a spike is accepted, no new spike can be detected for **1 millisecond**, preventing a single spike's falling/rising edge noise from being counted multiple times. This value is fixed and not user-adjustable.

4. **Bandpass filtering.** A 4th-order Butterworth filter (zero-phase, via forward-backward filtering) is applied before detection, with user-adjustable low/high cutoffs (default 300–3000 Hz).

---

## 8. Burst Detection Methods — Algorithm Details

### Max Interval (Cotterill et al. 2016)

Ranked #1 of 8 burst detection methods evaluated across 11 desirable properties in the source publication. A 5-parameter algorithm:

1. Scan the spike train's inter-spike intervals (ISIs). Whenever an ISI is ≤ the **max beginning ISI**, open a candidate burst.
2. Extend the candidate burst for as long as subsequent ISIs remain ≤ the **max end ISI**.
3. Merge two adjacent candidate bursts if the gap between them (the interburst interval) is shorter than the **min interburst interval**.
4. Keep only bursts meeting both the **min spikes per burst** and **min burst duration** criteria.

### logISI Adaptive (Pasquale et al. 2010)

A data-driven method that derives its own ISI threshold from the recording's own ISI distribution, rather than requiring the user to specify one:

1. Compute all ISIs and take their base-10 logarithm (in milliseconds).
2. Build a histogram in log-ISI space (fixed bin width = 0.1 log-units) and smooth it with a Gaussian kernel.
3. Identify the histogram's local maxima, splitting them into an **intra-burst peak** (ISI < 100 ms) and an **inter-burst peak** (ISI ≥ 100 ms).
4. Compute the **void parameter** between the two peaks:

   ```
   void = 1 − g(min) / √(g(peak₁) × g(peak₂))
   ```

   where `g(peak₁)`, `g(peak₂)` are the smoothed histogram heights at the two peaks, and `g(min)` is the smoothed height at the deepest point between them. A void parameter near 1 means the two peaks are well-separated (a clean bimodal ISI distribution); near 0 means they blend together.

5. If the void parameter meets or exceeds the **void parameter threshold** (default 0.7), the ISI threshold (`ISIth`) is set to the ISI value at that minimum point.
6. If `ISIth` exceeds 100 ms, **dual-threshold detection** is used: burst "cores" are found using a fixed 100 ms threshold, then each core's boundaries are extended outward using the more permissive `ISIth`. If `ISIth` is ≤ 100 ms, a single threshold at `ISIth` is used directly.
7. If no valid bimodal structure is found (void parameter too low, or too few ISIs to analyse), the method **falls back to a fixed 100 ms threshold** and flags this in its output.

### Both — compare methods

Runs both methods on the same spike train and quantifies their agreement using the **normalized Hamming distance** (as described in Cotterill et al. 2016): the recording is divided into 50 ms bins, each bin is marked "in a burst" or not according to each method independently, and the Hamming distance is the percentage of bins where the two methods disagree.

| Hamming distance | Interpretation |
|---|---|
| < 5% | Strong agreement — burst calls are robust |
| 5% – 10% | Moderate agreement — broadly consistent |
| > 10% | Poor agreement — review parameters or ISI structure |

---

## 9. Tab-by-Tab Reference

Every figure can be downloaded as a combined PNG (the full multi-panel figure as shown) **and**, where a figure has multiple panels, each individual panel can also be downloaded separately via a small "⬇️" button beneath the figure — useful when you only need one panel for a slide or figure, not the whole composite.

### Overview
Two panels: a **spike raster** (every detected/supplied spike plotted as a tick mark over time, with detected burst windows shaded and labeled), and the **instantaneous firing rate** over the same timeline. Gives an immediate visual sense of the recording's overall activity pattern and where bursts occur.

### Raw Trace
The continuous voltage trace, with a toggle to switch between the **raw (unfiltered)** and **bandpass-filtered** signal. Long traces are automatically downsampled for display (with the downsampling factor noted in the title) so the plot stays responsive; the underlying analysis always uses the full-resolution data regardless of the display downsampling. Use this tab to sanity-check the recording itself — confirm the electrode wasn't saturated or flat, and see what the bandpass filter removed versus preserved.

### ISI Analysis
Two histograms of inter-spike intervals — linear and log-scaled counts — colored by whether each ISI falls below the active burst-detection threshold. Reports mean ISI overall, mean intra-burst ISI, and mean inter-burst ISI.

### Amplitude
Four panels quantifying spike amplitude from the extracted waveforms:
- **All spike waveforms overlaid**, colored by trough magnitude, with the mean waveform highlighted and its peak-to-peak amplitude annotated.
- **Trough amplitude distribution** (histogram, with mean/median/threshold marked).
- **Peak-to-peak amplitude distribution** (histogram, with mean/median marked).
- **Amplitude over time** (scatter, with a linear trend line and its slope reported in µV/s) — a declining slope across a long recording can indicate electrode drift or declining cell health, an important caveat for any analysis assuming a stationary signal.

If the recording contains both burst and non-burst (isolated) spikes, an additional **in-burst vs. isolated amplitude** boxplot is shown — spikes fired during a burst are frequently smaller than isolated spikes due to short-term depression, and this comparison tests for that directly.

### Waveform Metrics
Three panels characterizing spike waveform shape:
- **Spike width** (trough-to-peak duration) histogram — a narrow, unimodal distribution suggests a homogeneous spike population; a wide or multimodal one can indicate noise contamination or a mix of distinct units that a single channel can't separate on its own.
- **Amplitude vs. spike width** scatter — a lightweight two-feature check for distinct spike populations (genuinely different units often separate visually on these two axes even without full spike sorting).
- **SNR distribution** histogram — shows how much of the detected spike population sits comfortably above the noise floor versus near the detection threshold, useful for judging whether the threshold is well-chosen.

### Bursts
Per-burst duration and spike count (bar chart), plus an in-burst vs. isolated amplitude boxplot. Below the figure, a data table reports per-burst statistics: **mean/max amplitude**, **amplitude standard deviation and coefficient of variation**, and the **attenuation index** (see [Section 10](#10-metrics--formulas-glossary)). A second figure, **Burst-Level Correlations**, plots five relationships across all detected bursts — duration vs. mean amplitude, duration vs. attenuation, spike count vs. attenuation, duration vs. mean spike width, and duration vs. spike count — each with a linear fit and Pearson correlation coefficient when enough bursts are present. These test whether burst-level properties are related (e.g. do longer bursts attenuate more?), a common analysis in burst-dynamics publications.

### Burst Amplitude Dynamics
Three panels:
- **Intra-burst amplitude decrement** — spike amplitude plotted against its position within its burst (1st, 2nd, 3rd spike, ...), aggregated across all bursts with a mean-per-position line overlaid. Reveals whether spikes systematically shrink (or grow) across the course of a burst.
- **Amplitude vs. preceding ISI** — tests whether spikes firing soon after the previous spike show reduced amplitude (a signature of incomplete recovery from the refractory period).
- **Amplitude vs. following ISI** — the complementary check, whether a spike's amplitude predicts the timing of the next spike.

### logISI Histogram
The log-scaled ISI histogram used by the logISI Adaptive method, showing the Gaussian-smoothed curve, the detected intra-burst and inter-burst peaks, and the resulting threshold (or the 100 ms fallback, if applicable). This is shown even when Max Interval is the active detection method, purely for inspection of the recording's ISI structure. When "Both" is selected, an additional **comparison raster** shows the Max Interval and logISI burst calls stacked over the same spike train, letting you visually inspect where the two methods agree or disagree.

### Data Table
A complete per-spike table: spike number, timestamp, trough amplitude, peak-to-peak amplitude, SNR, burst membership, spike width, and preceding/following ISI. Exportable as CSV for use in external statistical software.

---

## 10. Metrics & Formulas Glossary

| Metric | Formula | Interpretation |
|---|---|---|
| **Noise floor** | `median(\|filtered signal\|) / 0.6745` | Robust estimate of background noise amplitude, insensitive to spike outliers. |
| **SNR** | `\|spike trough\| / noise floor` | How far a spike's amplitude sits above the noise floor, in multiples of the noise level. |
| **Peak-to-peak amplitude** | `max(waveform) − min(waveform)` within the extracted spike window | The primary amplitude measure used throughout the tool. |
| **Spike width (trough-to-peak)** | Time from the waveform's trough to the following peak | A basic waveform-shape descriptor; helps flag noise contamination or mixed unit populations. |
| **Preceding / following ISI** | Time to the previous / next spike | Used to test amplitude-timing relationships (recovery/refractory effects). |
| **Burst attenuation index** | `(first spike amplitude − last spike amplitude) / first spike amplitude` | Positive = amplitude shrank across the burst; negative = amplitude grew. Zero = no net change. |
| **Coefficient of variation (CV)** | `standard deviation / mean` | A scale-independent measure of amplitude variability within a burst. |
| **Hamming distance (method agreement)** | `% of 50 ms bins where Max Interval and logISI disagree` | Quantifies how consistently the two burst-detection methods identify the same time windows as bursting. |

---

## 11. Exporting for Publication

### Methods Text generator
The "Generate Methods Text" button produces a ready-to-paste methods paragraph that adapts automatically to the burst detection method currently selected:

- **Max Interval selected:** reports the threshold multiplier, bandpass range, and refractory period used for spike detection (with the Quiroga et al. 2004 citation), then the Max Interval method's five parameter values (with the Cotterill et al. 2016 citation), and finally the waveform window used for amplitude quantification (with the Obien et al. 2015 citation).
- **logISI Adaptive selected:** as above, but reports the automatically-derived ISI threshold and void parameter, whether it exceeded 100 ms, and whether single- or dual-threshold detection was applied (noting explicitly if the method fell back to the fixed 100 ms criterion).
- **Both selected:** reports both methods' parameters/results, plus a sentence describing their agreement level and Hamming distance.

The generated paragraph is shown in a text box for you to copy directly into a manuscript.

### Figure export
Every figure can be downloaded as a single combined PNG. Multi-panel figures additionally offer per-panel PNG downloads (the small "⬇️" buttons beneath the figure), for when you need just one panel for a slide or manuscript figure rather than the full composite.

### Data export
The full per-spike Data Table can be downloaded as a CSV file, containing every spike's timestamp, amplitude, SNR, burst membership, spike width, and ISI values for further analysis outside the tool.

---

## 12. Known Limitations

- **Single-channel only.** The tool analyses one voltage trace at a time; there is currently no support for multi-electrode spatial analysis (e.g. per-electrode comparisons or array heatmaps).
- **Edge-trimming during waveform extraction.** Spikes occurring too close to the very start or end of the recording (closer than the pre-/post-spike window) are excluded from waveform-based metrics (amplitude, width, SNR, burst-amplitude-dynamics plots), since a full waveform window can't be extracted for them. This affects only a small number of spikes at the extreme edges of a recording.
- **logISI fallback.** If a recording's ISI distribution doesn't show a clear bimodal structure (or there are too few spikes to analyse), the logISI method automatically falls back to a fixed 100 ms threshold rather than failing outright — this is flagged in the logISI Histogram tab and in the generated methods text when it occurs.
- **Sampling rate is inferred, not manually set.** The sampling rate is calculated from the median spacing between timestamps in the uploaded file. An irregular or corrupted time column will produce an incorrect sampling rate and should be checked before relying on the analysis.
- **Insufficient-data messages.** Some tabs (e.g. Burst Amplitude Dynamics, Amplitude's in-burst comparison) will show an informational message instead of a plot if there are too few bursts or multi-spike bursts in the current recording/parameter combination to compute a meaningful result.

---

## 13. Glossary of Terms

- **ISI (Inter-Spike Interval)** — the time between two consecutive spikes.
- **MAD (Median Absolute Deviation)** — a robust measure of statistical spread, used here to estimate the noise floor without being skewed by spike amplitudes.
- **Void parameter** — a measure (0–1) of how well-separated the intra-burst and inter-burst peaks are in a log-ISI histogram; higher means a cleaner separation.
- **Refractory period** — a short time window after a detected spike during which no further spike can be detected, preventing double-counting of a single spike event.
- **Hamming distance** — here, the percentage of time bins in which two burst-detection methods disagree about whether a burst is occurring.
- **Attenuation index** — the fractional change in spike amplitude from the first to the last spike of a burst.
- **Peak-to-peak amplitude** — the voltage difference between a spike waveform's highest and lowest points.
- **SNR (Signal-to-Noise Ratio)** — here, a spike's amplitude expressed as a multiple of the estimated noise floor.

---

## 14. References

- Cotterill, E., et al. (2016). *A comparison of computational methods for detecting bursts in neuronal spike trains and their application to human stem cell-derived neuronal networks.* Journal of Neurophysiology, 116(2), 306–321.
- Pasquale, V., et al. (2010). *A self-adapting approach for the detection of bursts and network bursts in neuronal cultures.* Journal of Computational Neuroscience, 29(1–2), 213–229.
- Quiroga, R. Q., Nadasdy, Z., & Ben-Shaul, Y. (2004). *Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering.* Neural Computation, 16(8), 1661–1687.
- Obien, M. E. J., et al. (2015). *Revealing neuronal function through microelectrode array recordings.* Frontiers in Neuroscience, 8, 423.
