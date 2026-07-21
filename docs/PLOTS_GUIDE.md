# SpikeLab Plots & Metrics Guide

What each analysis added in this session shows, and why it's useful. Organized by tab.

---

## Raw Trace tab

### Raw / Filtered voltage trace (toggle)
**What:** The continuous voltage recording plotted over time, with a toggle to switch between the raw unfiltered signal and the bandpass-filtered signal used for spike detection.
**Why:** Lets you sanity-check the recording itself before trusting any downstream analysis — confirm the electrode wasn't saturated/flat, spot large artifacts, and see exactly what the bandpass filter removed (low-frequency drift, high-frequency noise) versus what it kept.

---

## Spike Waveforms tab

This tab replaces Raw Trace only for a standalone NeuroExplorer multichannel channel that contains bundled waveform samples.

### Supplied spike snippets and mean waveform
**What:** The selected electrode's pre-extracted 76-sample spike snippets overlaid on the confirmed 25 kHz, −1.0 to +2.0 ms time axis, together with the mean waveform and a ±1 standard-deviation band. For responsiveness, at most 500 evenly spaced snippets are drawn, while the mean and standard deviation always use every spike.
**Why:** Provides the closest scientifically valid trace view available for this input format. The export does not contain a continuous voltage recording, so the app cannot reconstruct raw and bandpass-filtered traces. These snippets are normalized to µV but are not filtered again.

---

## Amplitude tab

For standalone NeuroExplorer multichannel waveform text, plots use the selected electrode's 76-sample snippets on a confirmed 25 kHz, −1.0 to +2.0 ms axis. The source-unit selector defaults to mV and values are normalized to µV before any amplitude statistic or plot is calculated.

### Amplitude drift trend line (added to the existing "Spike Amplitude Over Time" panel)
**What:** A linear regression fit through peak-to-peak amplitude vs. time, with the slope reported in µV/s.
**Why:** A steadily declining slope over a long recording usually signals electrode drift, cell health decline, or gradual loss of contact — a scientifically important caveat for any analysis that assumes a stationary signal. The raw scatter alone makes this trend hard to eyeball; the fit quantifies it.

### In-Burst vs Isolated Amplitude (boxplot)
**What:** Peak-to-peak amplitude distributions split into two groups — spikes that fall inside a detected burst window vs. spikes firing in isolation.
**Why:** Spikes within bursts are frequently smaller than isolated spikes (due to short-term synaptic/channel depression during rapid firing). Comparing the two distributions is a standard check for this effect and can reveal whether your amplitude statistics are being skewed by burst activity.

---

## Waveform Metrics tab

### Spike Width (Trough-to-Peak) histogram
**What:** For each spike, the time (ms) from the waveform's trough to the following peak (repolarization), histogrammed across all spikes.
**Why:** Spike width is a basic waveform-shape descriptor. A narrow, unimodal width distribution suggests a homogeneous population of similar-looking spikes; a wide or multimodal distribution can indicate a mix of genuine units, noise contamination, or multi-unit activity that a single-channel electrode can't separate on its own.

### Amplitude vs Spike Width (scatter)
**What:** Peak-to-peak amplitude plotted against spike width, one point per spike.
**Why:** A cheap, two-feature "cluster check" — genuine, distinct unit populations often separate visually on amplitude and width even without full spike sorting. Tight, single-cloud data supports the assumption of one dominant unit; multiple visible clusters is a flag worth investigating.

### SNR Distribution histogram
**What:** Histogram of per-spike signal-to-noise ratio (|trough| ÷ noise floor).
**Why:** SNR was already computed per spike (used in the Data Table) but never visualized. The histogram shows how much of your spike population is comfortably above threshold vs. sitting near the noise floor — useful for judging whether the detection threshold is well-chosen, or whether a chunk of "spikes" are borderline noise.

SNR is only meaningful for a continuous voltage trace with an estimated noise floor. It is omitted for standalone pre-sorted waveform exports.

---

## Burst Amplitude Dynamics tab

### Intra-Burst Amplitude Decrement
**What:** Spike amplitude plotted against its position within its burst (1st, 2nd, 3rd spike, ...), aggregated across all bursts, with a mean-per-position line overlaid.
**Why:** A classic burst-dynamics plot. Many neurons show progressive amplitude attenuation across a burst (each successive spike smaller than the last) due to short-term depression — this plot makes that pattern visible and quantifiable across the whole recording rather than burst-by-burst.

### Amplitude vs Preceding ISI
**What:** Spike amplitude plotted against the time since the previous spike.
**Why:** Spikes that fire very soon after the previous one (short preceding ISI) often show reduced amplitude due to incomplete recovery from the refractory period. This scatter tests for that relationship directly.

### Amplitude vs Following ISI
**What:** Spike amplitude plotted against the time until the next spike.
**Why:** The complementary check — whether a spike's amplitude predicts how soon the next one arrives. Useful alongside the "preceding ISI" panel for a fuller picture of amplitude-timing coupling.

---

## Bursts tab

### Burst-wise amplitude stats (table columns: Mean/Max/SD/CV Amp, Attenuation Index)
**What:** Per-burst summary statistics — mean and max peak-to-peak amplitude within the burst, its standard deviation and coefficient of variation (SD ÷ mean), and an **attenuation index** defined as `(first spike amplitude − last spike amplitude) / first spike amplitude` (positive = amplitude shrank over the burst, negative = it grew).
**Why:** Turns the raw burst list into burst-level quantitative summaries suitable for reporting — e.g. "bursts showed a mean attenuation index of 0.31 ± 0.12," a standard way to characterize burst amplitude dynamics in publications.

### Burst-Level Correlations (5 scatter panels)
**What:** Burst duration vs. mean amplitude, burst duration vs. attenuation index, spike count vs. attenuation index, burst duration vs. mean spike width, and burst duration vs. spike count — each with a linear fit and Pearson r when enough bursts are available.
**Why:** Tests whether burst-level properties are related — e.g. do longer bursts attenuate more? Do bursts with more spikes show stronger attenuation? These relationships (or their absence) are often reported findings in burst-dynamics papers, not just descriptive stats.

---

## Data Table tab (new columns)

**Spike Width (ms), Preceding ISI (ms), Following ISI (ms)** — the same per-spike values used in the plots above, added to the exportable spike-level table so they're available for external analysis (e.g. in a stats package) beyond what's shown in the app's own plots.

Standalone multichannel imports also add **Electrode ID** and the complete **Source Label** to every CSV row so exported measurements retain their channel provenance.

---

## A note on the underlying amplitude measure

All "amplitude" values above are **peak-to-peak (p2p)** voltage, matching the convention already used elsewhere in the app (e.g. the Data Table's "Peak-to-Peak (µV)" column, the original Bursts-tab boxplot). Burst membership for all of the above is determined by matching each spike's timestamp against each burst's `[start, end]` time window — robust to the small number of spikes trimmed near the recording's edges during waveform extraction.
