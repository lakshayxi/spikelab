# SpikeLab — Documentation

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
9. [Analysis Workspace Reference](#9-analysis-workspace-reference)
10. [Metrics & Formulas Glossary](#10-metrics--formulas-glossary)
11. [Exporting for Publication](#11-exporting-for-publication)
12. [Known Limitations](#12-known-limitations)
13. [Glossary of Terms](#13-glossary-of-terms)
14. [References](#14-references)

---

## 1. Overview

The MEA Spike Analyser is an offline tool for analysing extracellular voltage recordings from Multi-Electrode Array (MEA) experiments. Given one active signal or pre-sorted electrode at a time, it detects or imports spikes, identifies bursts using two independent peer-reviewed algorithms, quantifies spike amplitude and waveform shape when waveforms are available, characterizes burst-level amplitude dynamics, and produces publication-ready figures, a spike-level data table, and a draft methods paragraph.

**Scope note:** the tool analyses **one selected channel/electrode at a time**. EDF and standalone NeuroExplorer multichannel exports provide selectors, but selecting a channel does not run batch or network analysis. There are no electrode-array heatmaps or cross-channel comparisons. A continuous text trace may still be reconstructed from several sequential export segments (see [Section 4](#4-input-file-formats)).

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
1. **Choose one primary input:** continuous two-column text/CSV, one EDF recording, or one standalone NeuroExplorer multichannel waveform text export.
2. **Select one signal/electrode** when the EDF or standalone export contains several populated channels.
3. **For continuous recordings only, optionally upload a NeuroExplorer spike-time overlay.** Its "Instantaneous Parameters" timestamps replace raw-signal detections downstream.
4. **For a standalone waveform export, confirm waveform unit and duration.** Waveform values default to mV and are normalized to µV. Duration defaults to the latest timestamp across every populated channel and may only be increased.
5. **Choose a burst detection method** in the sidebar: Max Interval, Adaptive logISI, or Compare both.
6. **Review and export results** — figures, spike-level CSV, and generated methods text all reflect the selected channel.

---

## 4. Input File Formats

### Continuous raw voltage text/CSV

A two-column file — **time (seconds)** and **voltage (µV)** — tab- or comma-separated, as exported from Multi Channel Analyzer as **"Raw Data Time Points."** Typical export files include a short header (software name/version, channel label, column units) before the data rows; this header is detected and skipped automatically, no manual editing is required. Example data rows:

```
1.358     -1.192096
1.35804   -0.596048
1.35808   2.682216
```

The file uploader accepts `.txt` and `.csv`. The delimiter (tab or comma) is detected from the first row whose first field looks numeric, so commas or tabs in exporter metadata do not determine the format; tab takes priority if that candidate row contains both. Blank lines and pre-data header or metadata rows whose first field is nonnumeric are ignored. From the first numeric-looking row onward, every nonblank row must contain finite numeric time and voltage values in the first two columns. Missing or malformed values are rejected with the filename and physical line number instead of being silently skipped. Additional columns remain supported and are ignored.

### Multi-segment upload & auto-stitching

Multi Channel Analyzer typically caps a single raw-trace export at around 10–12 seconds, so a full recording is often exported as several sequential files (e.g. `L5_1.txt`, `L5_2.txt`, `L5_3.txt`, ...). **Upload all segment files together** in the raw-trace uploader (it accepts multiple files at once) and the tool will:

1. **Reorder them by their own first timestamp** — regardless of the order you selected/uploaded them in, or how they're named. This means files can be uploaded in any order and will still be assembled correctly.
2. **Trim overlapping time regions.** Real export segments frequently overlap slightly at their boundaries (the end of one file's timestamps and the start of the next file's timestamps cover the same moment in time). The tool detects this automatically and keeps the earlier segment's data for any overlapping span, only appending the newer segment's genuinely new (later) samples — this avoids duplicated or aliased samples in the combined trace.
3. **Show a stitch summary.** After upload, an expandable panel reports the order the segments were assembled in, each segment's start/end time, how many samples were trimmed for overlap at each boundary, and the final combined recording duration — so you can visually verify the reconstruction is correct before trusting the analysis.

A single uploaded file skips all of this and is used directly, with identical behavior to previous versions of the tool.

### EDF recording

Upload one `.edf` file at a time. Ordinary signals are listed with their source label, independent sampling rate, sample count, and EDF-declared physical dimension. The selected signal's own digital/physical calibration is applied, then recognized voltage dimensions (`V`, `mV`, `uV`/`µV`, and common written variants) are normalized to **µV** before filtering and analysis. Annotation signals are not selectable.

EDF uploads are limited to **512 MiB (536,870,912 bytes)**. EDF processing currently occurs in memory: the uploaded content is materialized before parsing, and selected-channel extraction, voltage calibration, timestamp generation, and filtering require additional arrays. The upload limit is therefore a supported maximum, not a guarantee that every file at the limit will fit on a particular computer. Actual usable size also depends on available RAM and the selected signal's sample count/length. If processing runs out of memory, export a shorter recording or a file containing a smaller channel subset.

The importer rejects unsupported physical dimensions rather than assuming they are voltage. It also rejects truncated data records and **EDF+D** discontinuous recordings. EDF+D record start times can contain gaps, so joining their sample blocks into a regular time axis would be scientifically incorrect. Continuous EDF and EDF+C recordings are supported.

Large-file streaming and memory mapping are not implemented. Selecting one channel limits the signal materialized for analysis, but it does not turn the EDF path into a streaming workflow.

### Standalone NeuroExplorer multichannel waveform text

This is a quoted-header numeric export containing many electrode/unit groups, each with a `Spike_timestamps` column and, for populated waveform variables, exactly 76 `Spike_value_1` through `Spike_value_76` columns. It is a standalone primary input—not a continuous raw trace and not the optional overlay below—and cannot be combined with raw text segments or EDF.

The format uses **one literal ASCII space per field boundary** and reserves an additional empty spacer position for timestamp-only channel columns. Consecutive spaces are therefore meaningful placeholders, while each 77-column waveform group retains its full positional width even after it runs out of spikes. The parser derives those data widths from the header, so timestamps and waveform values cannot migrate between equal-width channels when one channel ends before another. It rejects partial waveform rows or headers, channels that resume after becoming blank, invalid/non-finite numbers, and non-increasing timestamps with the physical row number in the error.

Only populated channels are listed. The selector shows the concise electrode ID (for example, `A4`), complete original source label, spike count, and waveform availability. One selected electrode feeds the existing single-channel analysis; the app does not calculate network statistics.

Waveform exports do not encode amplitude units. The unit selector therefore defaults to **mV**, also offering µV and V, and all selected-channel waveforms are normalized to µV. These files use the confirmed **25 kHz** waveform sampling rate with 76 samples spanning **−1.0 ms through +2.0 ms** (25 pre-spike samples). The recording duration defaults to the latest timestamp across all populated channels—not merely the selected channel's spike span. It is editable upward, but cannot be shorter than the data. This duration is used for mean firing rate and the 50 ms-bin burst-occupancy comparison.

When the selected channel contains waveform samples, the **Signal → Spike waveforms** view overlays the supplied snippets and shows the all-spike mean with a ±1 standard-deviation band. This is not a reconstructed continuous trace: the export contains no continuous raw voltage, so there is no raw-versus-filtered toggle and the snippets are not filtered again.

### NeuroExplorer spike-time overlay (optional)

A standard NeuroExplorer ASCII export containing an **"Instantaneous Parameters"** section (spike time, ISI, and instantaneous firing rate columns). If uploaded, its spike timestamps are used for *all* downstream burst/amplitude/waveform analysis instead of the raw-signal spike detector — useful when you've already curated spike times in NeuroExplorer and want this tool's burst/amplitude analyses applied to that curated spike list. Note: the raw-signal spike detector still runs in the background regardless (to establish the noise floor and SNR reference), but its detected spike *times* are only used when no NeuroExplorer file is supplied.

This optional file can accompany continuous text/CSV or EDF only. It cannot accompany the standalone multichannel waveform export, which already contains pre-sorted timestamps. Files missing the "Instantaneous Parameters" section parse to an empty spike list; the app warns and falls back to raw detection.

---

## 5. The Analysis Pipeline

The pipeline branches according to the primary input:

1. **Continuous text/CSV or EDF:** parse one regular voltage signal in µV; stitch text segments when needed; infer the text/CSV sampling rate from timestamps or use the selected EDF channel's exact declared rate; bandpass filter; detect negative-going spikes; optionally substitute overlay timestamps; extract filtered waveforms.
2. **Standalone multichannel waveform text:** positionally parse every electrode, select one populated channel, normalize its bundled waveform values to µV, and use its pre-sorted timestamps directly. There is no filtering, raw threshold, noise floor, or SNR for this path.
3. **Shared analysis:** run the selected burst detector(s), derive available amplitude/ISI/burst metrics, render plots and tables, and generate channel-aware exports.

---

## 6. Sidebar Parameter Reference

### Spike Detection

| Parameter | Range | Default | Step | What it controls |
|---|---|---|---|---|
| Threshold multiplier (× σ) | 3.0 – 10.0 | 5.0 | 0.5 | Spike detection threshold, as a multiple of the estimated noise floor. Standard is 5×; higher values are more conservative (fewer, higher-confidence spikes). |
| Bandpass low (Hz) | 100 – 1000 | 300 | 50 | Low cutoff of the bandpass filter applied before spike detection. |
| Bandpass high (Hz) | 1000 – 6000 | 3000 | 100 | High cutoff of the bandpass filter. |

### Burst Detection Method

A single selector chooses which method(s) run: **Max Interval (Cotterill et al. 2016)**, **Adaptive logISI (Pasquale et al. 2010)**, or **Compare both**. Selecting a method reveals only its relevant parameters below.

**Max Interval parameters** (shown for Max Interval or Compare both):

| Parameter | Range | Default | Step | Meaning |
|---|---|---|---|---|
| Max beginning ISI (ms) | 50 – 500 | 170 | 10 | Longest inter-spike interval that can *open* a new burst. |
| Max end ISI (ms) | 100 – 1000 | 300 | 10 | Longest inter-spike interval that can *continue* an already-open burst. |
| Min interburst interval (ms) | 50 – 1000 | 200 | 10 | Minimum gap required between two candidate bursts for them to be counted as separate events (shorter gaps cause them to merge). |
| Min burst duration (ms) | 5 – 200 | 10 | 5 | Shortest event duration that still counts as a burst. |

**logISI parameters** (shown for Adaptive logISI or Compare both):

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

These sidebar windows apply to continuous voltage traces. Standalone NeuroExplorer waveform exports already contain a fixed 76-sample, −1.0 to +2.0 ms window and do not use these controls.

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

### Adaptive logISI (Pasquale et al. 2010)

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

### Compare both

Runs both methods on the same spike train and compares their burst occupancy using the **normalized Hamming distance** (as described in Cotterill et al. 2016): the recording is divided into 50 ms bins, each bin is marked "in a burst" or not according to each method independently, and the Hamming distance is the percentage of bins where the two methods disagree.

| Hamming distance | Interpretation |
|---|---|
| < 5% | High agreement on burst occupancy |
| 5% – 10% | Moderate agreement on burst occupancy |
| > 10% | Low agreement on burst occupancy — review parameters or ISI structure |

---

## 9. Analysis Workspace Reference

Results are grouped into five top-level workspaces: **Overview**, **Signal**, **Spike analysis**,
**Burst analysis**, and **Data & export**. The latter four use a compact view selector when several
subviews are available; when only one view applies, its content is shown directly.

Each figure places its download control in the top-right of the figure header. A single-panel figure
uses a direct **Download PNG** button. A multi-panel figure uses a **Download** menu containing the
combined image and individual panel crops. The amplitude download menu also includes its related
burst-membership comparison when that figure is available.

For responsiveness on spike-rich recordings, waveform overlay panels draw at most **500**
deterministic, approximately evenly spaced representative waveforms, and large scatter panels draw
at most **5,000** representative points. A caption reports whenever a visual subset is shown. This
affects rendering only: numerical analysis, histograms, means, trends, correlations, burst
detection, metrics, and exports continue to use all valid data.

### Overview

The overview contains a spike raster and instantaneous firing rate over a shared time axis. Detected
burst windows are shaded and labelled. The full recording is shown by default; clear **Time window**
controls allow a smaller interval to be inspected without rerunning the analysis.

### Signal

#### Raw trace

Available for continuous text/CSV and EDF inputs. A toggle switches between the raw and
bandpass-filtered voltage signal. Long traces are downsampled for display responsiveness, with the
rendering detail reported below the figure; analysis always uses the full-resolution arrays.

#### Spike waveforms

For continuous recordings, this view shows waveform snippets extracted from the filtered trace
around detected or overlaid spike timestamps. For a standalone NeuroExplorer export, it shows the
selected electrode's supplied 76-sample snippets on the confirmed 25 kHz, −1.0 to +2.0 ms axis
after source-unit normalization to µV. Supplied NeuroExplorer snippets are not filtered again.

The overlay draws at most 500 representative snippets, while the mean and ±1 standard-deviation
summary use all available waveforms.

### Spike analysis

#### Amplitude

The amplitude view reports mean trough, mean peak-to-peak amplitude, peak-to-peak standard
deviation, and SNR where a continuous noise reference exists. Its figure contains waveform,
distribution, and amplitude-over-time panels. When both burst and isolated spikes are available, a
related burst-membership comparison is also shown.

#### Waveform metrics

This view presents the trough-to-peak spike-width distribution, amplitude versus width, and SNR
distribution where applicable. SNR is unavailable for standalone pre-sorted waveform exports
because they contain no raw noise trace or detection threshold.

#### ISI analysis

Linear and log-scaled inter-spike interval histograms are coloured relative to the active
burst-detection threshold. Summary metrics report overall, intra-burst, and inter-burst ISI.

### Burst analysis

#### Bursts

The primary burst view reports burst count, mean duration, mean spikes per burst, and mean
interburst interval. It visualizes duration, spike count, and amplitude membership, followed by
burst-level correlation panels when enough data are available.

#### Burst dynamics

This view combines intra-burst amplitude decrement with amplitude versus preceding and following
ISI. It is shown only when the selected data contain enough waveform-aligned burst or ISI
measurements.

#### Adaptive logISI

The histogram shows the smoothed logISI distribution, detected modes, void parameter, and adaptive
or fallback threshold. In Max Interval mode it is explicitly presented for inspection only. In
**Compare both** mode, a comparison raster also shows Max Interval and logISI burst occupancy over
the same spike train.

### Data & export

#### Spike data

The waveform-aligned spike table contains timestamp, trough, peak-to-peak amplitude, SNR where
applicable, burst membership, width, and neighbouring ISIs. Standalone multichannel rows also retain
the concise electrode ID and complete source label.

#### Burst data

The burst table contains timing, duration, spike count, amplitude statistics, attenuation, mean
width, and the active burst-detection method. In **Compare both** mode, shared displays and exports
use Max Interval as the primary burst set.

#### Exports & methods

This view downloads spike CSV when waveform-aligned rows are available, burst CSV, and analysis
metadata JSON. It also generates a draft methods paragraph from the active input route and analysis
settings; every generated paragraph carries a publication-review warning.

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
| **Hamming distance (burst occupancy)** | `% of 50 ms bins where Max Interval and logISI disagree` | Measures how often the methods assign different burst-occupancy states to the same time bins. |

---

## 11. Exporting for Publication

### Methods Text generator
The "Generate Methods Text" button produces a ready-to-paste methods paragraph that adapts automatically to the burst detection method currently selected:

- **Max Interval selected:** reports the threshold multiplier, bandpass range, and refractory period used for spike detection (with the Quiroga et al. 2004 citation), then the Max Interval method's five parameter values (with the Cotterill et al. 2016 citation), and finally the waveform window used for amplitude quantification (with the Obien et al. 2015 citation).
- **Adaptive logISI selected:** as above, but reports the automatically-derived ISI threshold and void parameter, whether it exceeded 100 ms, and whether single- or dual-threshold detection was applied (noting explicitly if the method fell back to the fixed 100 ms criterion).
- **Compare both selected:** reports both methods' parameters/results, plus the high/moderate/low agreement level on burst occupancy and the Hamming distance.
- **Standalone multichannel waveform input:** records the concise electrode ID, original source label, editable recording duration, 25 kHz/−1.0 to +2.0 ms waveform timing, selected source amplitude unit, and normalization to µV.

The generated paragraph is shown in a text box for you to copy into a manuscript. It is accompanied by a warning that it is a draft based on the current settings and that all parameters, citations, and experimental details must be verified before publication.

### Figure export
Every figure places its download control in the top-right of the figure header. Single-panel figures use a direct **Download PNG** button. Multi-panel figures use a **Download** menu containing the combined PNG and individual panel crops, for when you need just one panel for a slide or manuscript figure.

### Data export
When waveform-aligned spike rows are available, the Spike data table can be downloaded as `spike_data.csv`, containing each valid waveform's spike timestamp, amplitude, SNR where applicable, burst membership, spike width, and ISI values. Standalone multichannel exports additionally include electrode ID and the unmodified source label.

The separate `burst_data.csv` contains the stable burst-level fields described above. It remains available with column headers even when no bursts were detected.

`analysis_metadata.json` records the application/schema version, UTC analysis time, source files and input route, selected channel where applicable, recording bounds and sampling rate, spike source and counts, burst method/results, active parameters, optional logISI/comparison results, waveform unit, and available parser/stitch details. Numeric values are emitted as standard JSON values, while unavailable or non-finite values are `null`. It intentionally excludes raw traces, waveform arrays, file checksums, dependency versions, and bundled/ZIP exports.

---

## 12. Known Limitations

- **One selected channel at a time.** Multichannel EDF and NeuroExplorer files can be opened, but the tool analyses only the selected signal/electrode. There is no batch, network, spatial, or cross-electrode analysis.
- **Standalone waveform metadata.** NeuroExplorer waveform text does not encode amplitude units or full acquisition duration. Confirm the unit selector (default mV) and increase the duration if acquisition continued beyond the latest spike.
- **EDF memory and file size.** EDF uploads are limited to 512 MiB and processed in memory. Parsing, calibration, timestamps, and filtering allocate additional arrays, so the practical maximum depends on available RAM and selected-signal length and may be lower. Large-file streaming and memory mapping are not yet implemented.
- **Discontinuous EDF.** EDF+D is intentionally rejected because its discontinuous record timing cannot be represented by the app's regular continuous-trace pipeline.
- **Edge-trimming during waveform extraction.** Spikes occurring too close to the very start or end of the recording (closer than the pre-/post-spike window) are excluded from waveform-based metrics (amplitude, width, SNR, burst-amplitude-dynamics plots), since a full waveform window can't be extracted for them. This affects only a small number of spikes at the extreme edges of a recording.
- **logISI fallback.** If a recording's ISI distribution doesn't show a clear bimodal structure (or there are too few spikes to analyse), the logISI method automatically falls back to a fixed 100 ms threshold rather than failing outright — this is flagged in the logISI Histogram tab and in the generated methods text when it occurs.
- **Sampling rate handling.** Continuous text sampling rate is inferred as the rounded reciprocal of the median timestamp spacing. Every interval must be within ±5% of that median (apart from a machine-precision comparison floor); missing samples, acquisition gaps, and more irregular timestamps are rejected with the median period, largest interval, and irregular-interval count. EDF preserves the selected signal's exact rate from its declared samples-per-record and record duration. Standalone waveform text uses the confirmed fixed 25 kHz snippet rate.
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
