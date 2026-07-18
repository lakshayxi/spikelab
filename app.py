import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from parsers import (  # noqa: E402
    edf_signal_labels,
    is_ne_multichannel_content,
    parse_edf_content,
    parse_ne_multichannel_content,
    parse_neuroexplorer_content,
    parse_raw_content,
    stitch_segments,
)
from plots import (  # noqa: E402
    C_MUTED,
    C_SPIKE,
    _strip_emoji,
    fig_to_bytes,
    plot_amplitude,
    plot_amplitude_burst_membership,
    plot_burst_amplitude_dynamics,
    plot_burst_correlations,
    plot_bursts,
    plot_comparison_raster,
    plot_isi,
    plot_logisi_histogram,
    plot_overview,
    plot_raw_trace,
    plot_waveform_metrics,
    render_panel_downloads,
)
from processing import (  # noqa: E402
    bandpass_filter,
    build_summary_df,
    burst_membership_mask,
    compare_methods,
    compute_burst_amplitude_stats,
    compute_intraburst_decrement,
    compute_isi_arrays,
    compute_spike_widths,
    detect_spikes,
    extract_waveforms,
    logisi_method,
    max_interval_method,
    waveform_amplitude_stats,
)

# Sampling rate for NeuroExplorer multi-channel waveform snippets — this format
# carries no per-file timing metadata for Spike_value_1..N samples; 25 kHz is the
# recording system's known rate, confirmed for this lab's exports.
NE_MULTICHANNEL_WAVEFORM_FS = 25_000

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MEA Spike Analyser",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Cached parsing wrappers ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def parse_raw_file(content: bytes):
    return parse_raw_content(content)

@st.cache_data(show_spinner=False)
def parse_neuroexplorer_file(content: bytes):
    return parse_neuroexplorer_content(content)

# Keyed on file identity — the leading-underscore `_content` is not hashed, so
# switching the selected channel never re-runs the whole-file reconciliation.
@st.cache_data(show_spinner=False)
def parse_ne_multichannel_file(file_key: str, _content: bytes):
    return parse_ne_multichannel_content(_content)

# Read the upload's bytes once per file (leading-underscore `_uploaded` is not
# hashed), so a multi-GB EDF isn't re-copied into memory on every rerun.
@st.cache_data(show_spinner=False)
def _read_upload_bytes(file_key: str, _uploaded):
    return _uploaded.getvalue()

# Keyed on (file identity, channel) — the leading-underscore `_content` is not
# hashed, so switching channels or nudging a slider never re-hashes a multi-GB EDF.
@st.cache_data(show_spinner=False)
def parse_edf_file(file_key: str, channel_index: int, _content: bytes):
    return parse_edf_content(_content, channel_index)

# ── Cached burst wrappers (spike_times as tuple for cache hashing) ────────────
@st.cache_data(show_spinner=False)
def _cached_mi(spike_t, beg, end, ibi, dur, spk):
    return max_interval_method(np.array(spike_t), beg, end, ibi, dur, spk)

@st.cache_data(show_spinner=False)
def _cached_logisi(spike_t, min_spk, void_th):
    return logisi_method(np.array(spike_t), min_spk, void_th)

def _segment_validation_errors(t, v, label):
    errors = []
    if len(t) == 0 or len(v) == 0:
        return [f"{label}: no numeric time/voltage rows were found."]
    if len(t) != len(v):
        errors.append(f"{label}: time and voltage columns have different lengths.")
    if len(t) < 2:
        errors.append(f"{label}: at least two samples are required.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)):
        errors.append(f"{label}: contains non-finite values.")
    if len(t) >= 2 and not np.all(np.diff(t) > 0):
        errors.append(f"{label}: timestamps must be strictly increasing.")
    return errors

def _render_figure(fig, filename, panels=None, base_filename=None, combined=True):
    st.pyplot(fig, use_container_width=True)
    label = "Download PNG (combined)" if combined else "Download PNG"
    st.download_button(label, fig_to_bytes(fig), filename, "image/png")
    if panels and base_filename:
        render_panel_downloads(fig, panels, base_filename)
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.title("MEA Spike Analyser")
st.markdown("Spike detection, burst analysis and amplitude quantification for Multi-Electrode Array recordings.")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")

    st.markdown("**Spike Detection**")
    thr_mult = st.slider("Threshold multiplier (× σ)", 3.0, 10.0, 5.0, 0.5,
                         help="Standard is 5×. Higher values are more conservative (fewer spikes).")
    bp_low   = st.number_input("Bandpass low (Hz)",  100, 1000, 300, 50)
    bp_high  = st.number_input("Bandpass high (Hz)", 1000, 6000, 3000, 100)

    st.markdown("---")
    st.markdown("**Burst Detection Method**")
    burst_method = st.radio(
        "Method",
        options=[
            "Max Interval (Cotterill et al. 2016)",
            "logISI Adaptive (Pasquale et al. 2010)",
            "Both — compare methods",
        ],
        label_visibility="collapsed",
    )

    show_mi     = burst_method != "logISI Adaptive (Pasquale et al. 2010)"
    show_logisi = burst_method != "Max Interval (Cotterill et al. 2016)"

    if show_mi:
        st.markdown("*Max Interval parameters*")
        max_beg_isi = st.slider("Max beginning ISI (ms)",  50,  500, 170, 10)
        max_end_isi = st.slider("Max end ISI (ms)",        100, 1000, 300, 10)
        min_ibi_ms  = st.slider("Min interburst interval (ms)", 50, 1000, 200, 10)
        min_dur_ms  = st.slider("Min burst duration (ms)",  5, 200, 10, 5)
        st.caption("Cotterill et al. (2016) J Neurophysiol — ranked #1 of 8 burst detection algorithms")
    else:
        max_beg_isi = max_end_isi = min_ibi_ms = min_dur_ms = None

    if show_mi and show_logisi:
        st.markdown("---")

    if show_logisi:
        st.markdown("*logISI parameters*")
        void_thresh = st.slider("Void parameter threshold", 0.0, 1.0, 0.7, 0.05)
        st.caption("Pasquale et al. (2010) J Comput Neurosci — self-adapting, data-driven threshold")
    else:
        void_thresh = 0.7

    min_spk = st.slider("Min spikes per burst", 2, 10, 3, 1)

    st.markdown("---")
    st.markdown("**Waveform Extraction**")
    pre_ms  = st.slider("Pre-spike window (ms)",  0.5, 3.0, 1.0, 0.25)
    post_ms = st.slider("Post-spike window (ms)", 1.0, 5.0, 2.0, 0.25)

# ── File upload ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    raw_files = st.file_uploader(
        "Recording or standalone waveform export (.txt, .csv or .edf)",
        type=['txt', 'csv', 'edf'],
        accept_multiple_files=True,
        help="Continuous text/CSV: two columns (time in s, voltage in µV); multiple sequential "
             "segments may be stitched. EDF: one continuous recording with a signal selector. "
             "Standalone NeuroExplorer waveform text: one multichannel export uploaded alone; "
             "the selected electrode's pre-sorted spikes are analysed without a raw trace."
    )
with col2:
    ne_file = st.file_uploader(
        "Optional NeuroExplorer spike-time overlay (.txt)",
        type=['txt'],
        help="For a continuous text/CSV or EDF recording only: replaces detected spike times "
             "using a NeuroExplorer 'Instantaneous Parameters' export. This is different from "
             "the standalone multichannel waveform export accepted on the left."
    )

if not raw_files:
    st.info(
        "Upload one of three primary inputs: a continuous two-column **text/CSV** trace "
        "(time in s, voltage in µV; sequential segments may be stitched), one continuous "
        "**EDF** recording, or one standalone **NeuroExplorer multichannel waveform text** "
        "export. The standalone export supplies pre-sorted spikes and is not a raw trace."
    )
    st.stop()

# ── Route by format: EDF / multi-channel NeuroExplorer / text-CSV segments ────
edf_files = [f for f in raw_files if f.name.lower().endswith('.edf')]
txt_like_files = [f for f in raw_files if not f.name.lower().endswith('.edf')]

ne_mc_files, txt_files = [], []
for f in txt_like_files:
    peek = f.read()
    f.seek(0)
    (ne_mc_files if is_ne_multichannel_content(peek) else txt_files).append(f)

stitch_summary = None
has_raw_trace = not ne_mc_files
noise_floor = threshold = np.nan

if edf_files and (txt_files or ne_mc_files):
    st.error("Upload an EDF recording on its own — it can't be combined with text/CSV segments "
              "or a multi-channel NeuroExplorer export.")
    st.stop()

if ne_mc_files and txt_files:
    st.error("Upload a multi-channel NeuroExplorer export on its own — it can't be combined "
              "with plain text/CSV segments.")
    st.stop()

if ne_mc_files and ne_file:
    st.error(
        "The optional NeuroExplorer spike-time overlay cannot be combined with a standalone "
        "multichannel waveform export; that export already supplies its own spike timestamps."
    )
    st.stop()

if ne_mc_files:
    if len(ne_mc_files) > 1:
        st.error("Please analyse one multi-channel NeuroExplorer export at a time.")
        st.stop()
    ne_mc_file = ne_mc_files[0]
    file_key = getattr(ne_mc_file, 'file_id', None) or f"{ne_mc_file.name}:{getattr(ne_mc_file, 'size', 0)}"
    ne_mc_bytes = _read_upload_bytes(file_key, ne_mc_file)
    with st.spinner("Reconstructing channels from the NeuroExplorer export..."):
        try:
            ne_mc_channels = parse_ne_multichannel_file(file_key, ne_mc_bytes)
        except ValueError as exc:
            st.error(f"Could not read the multi-channel NeuroExplorer export: {exc}")
            st.stop()
    if not ne_mc_channels:
        st.error("This export has no channels with any spike data.")
        st.stop()

    sel = st.selectbox(
        "NeuroExplorer channel",
        options=list(range(len(ne_mc_channels))),
        format_func=lambda k: (
            f"{ne_mc_channels[k]['channel_id']} — {ne_mc_channels[k]['label']} — "
            f"{len(ne_mc_channels[k]['spike_times'])} spikes"
            + (" (with waveforms)" if ne_mc_channels[k]['waveforms'] is not None else "")
        ),
        help="Only populated channels are listed; one selected electrode is analysed at a time.",
    )
    ne_mc_channel = ne_mc_channels[sel]
    analysis_spikes = ne_mc_channel['spike_times']
    ne_mc_waveforms_native = ne_mc_channel['waveforms']
    if ne_mc_waveforms_native is not None:
        waveform_unit = st.selectbox(
            "Waveform amplitude unit",
            options=["mV", "µV", "V"],
            index=0,
            help="NeuroExplorer does not encode the waveform unit in this export. "
                 "Choose the source unit; amplitudes are normalized to µV for analysis and export.",
        )
        waveform_to_uv = {"V": 1_000_000.0, "mV": 1_000.0, "µV": 1.0}[waveform_unit]
        ne_mc_waveforms = ne_mc_waveforms_native * waveform_to_uv
    else:
        waveform_unit = None
        ne_mc_waveforms = None

    latest_timestamp = max(float(channel['spike_times'][-1]) for channel in ne_mc_channels)
    rec_dur = st.number_input(
        "Recording duration (s)",
        min_value=latest_timestamp,
        value=latest_timestamp,
        step=1.0,
        format="%.3f",
        help="Inferred from the latest timestamp across every populated electrode. "
             "Increase it if the acquisition continued after the final spike; it cannot "
             "be shorter than the data.",
    )
elif edf_files:
    if len(edf_files) > 1:
        st.error("Please analyse one EDF recording at a time.")
        st.stop()
    edf_file = edf_files[0]
    file_key = getattr(edf_file, 'file_id', None) or f"{edf_file.name}:{getattr(edf_file, 'size', 0)}"
    edf_bytes = _read_upload_bytes(file_key, edf_file)
    try:
        edf_channels = edf_signal_labels(edf_bytes)
    except ValueError as exc:
        st.error(f"Could not read the EDF file: {exc}")
        st.stop()
    if not edf_channels:
        st.error("The EDF file contains no ordinary signal channels to analyse.")
        st.stop()

    sel = st.selectbox(
        "EDF signal channel",
        options=list(range(len(edf_channels))),
        format_func=lambda k: (
            f"{edf_channels[k][1]} — {edf_channels[k][2]:g} Hz, "
            f"{edf_channels[k][3]:,} samples — declared {edf_channels[k][4] or '(blank)'}"
        ),
        help="One selected EDF signal is analysed at a time; its declared unit is shown here.",
    )
    channel_index = edf_channels[sel][0]

    with st.spinner("Reading EDF signal..."):
        try:
            raw_t, raw_v = parse_edf_file(file_key, channel_index, edf_bytes)
        except ValueError as exc:
            st.error(f"Could not read the selected EDF channel: {exc}")
            st.stop()
    validation_errors = _segment_validation_errors(raw_t, raw_v, edf_file.name)
    if validation_errors:
        st.error("The EDF channel could not be analysed:\n\n" + "\n".join(f"- {e}" for e in validation_errors))
        st.stop()
else:
    with st.spinner("Loading signal..."):
        segments = []
        validation_errors = []
        for f in txt_files:
            t, v = parse_raw_file(f.read())
            validation_errors.extend(_segment_validation_errors(t, v, f.name))
            segments.append((t, v))
        if validation_errors:
            st.error("The uploaded raw trace could not be analysed:\n\n" + "\n".join(f"- {e}" for e in validation_errors))
            st.stop()
        if len(segments) == 1:
            raw_t, raw_v = segments[0]
        else:
            raw_t, raw_v, stitch_summary = stitch_segments(segments)

ne_spike_times, ne_isis, ne_freqs = None, None, None

if has_raw_trace:
    # ── Infer sampling rate & filter (shared by EDF and text/CSV input) ────────
    with st.spinner("Filtering signal..."):
        sample_diffs = np.diff(raw_t)
        if len(sample_diffs) == 0 or not np.all(sample_diffs > 0):
            st.error("The trace timestamps must be strictly increasing.")
            st.stop()
        fs = round(1.0 / np.median(sample_diffs))
        try:
            raw_filt = bandpass_filter(raw_v, fs, bp_low, bp_high)
        except ValueError as exc:
            st.error(f"Could not filter the uploaded trace: {exc}")
            st.stop()

    if stitch_summary is not None:
        with st.expander(
            f"Stitched {len(txt_files)} segments into one "
            f"{raw_t[-1] - raw_t[0]:.1f}s trace", expanded=False
        ):
            st.dataframe(pd.DataFrame([{
                'Order':                      k + 1,
                'File':                       txt_files[row['segment_index']].name,
                'Start (s)':                  round(row['first_t'], 4),
                'End (s)':                    round(row['last_t'], 4),
                'Samples':                    row['n_samples'],
                'Overlap trimmed (s)':        round(row['overlap_s'], 4),
                'Overlap trimmed (samples)':  row['overlap_samples'],
                'Samples kept':               row['n_kept'],
            } for k, row in enumerate(stitch_summary)]), use_container_width=True, hide_index=True)

    with st.spinner("Detecting spikes..."):
        spike_times, spike_idxs, threshold, noise_floor = detect_spikes(
            raw_t, raw_filt, thr_mult, fs
        )

    if ne_file:
        ne_spike_times, ne_isis, ne_freqs = parse_neuroexplorer_file(ne_file.read())
        if len(ne_spike_times) == 0:
            st.warning(
                "The NeuroExplorer file did not contain any spike timestamps in an "
                "'Instantaneous Parameters' section, so raw-signal spike detection will be used instead."
            )
            ne_spike_times, ne_isis, ne_freqs = None, None, None

    analysis_spikes = ne_spike_times if ne_spike_times is not None else spike_times
    rec_dur = float(raw_t[-1] - raw_t[0])
else:
    # The standalone multichannel export's editable duration was initialized
    # from the latest timestamp across all populated channels during routing.
    rec_dur = float(rec_dur)

if ne_freqs is not None:
    analysis_freqs = ne_freqs
else:
    # No NeuroExplorer rates: derive the instantaneous firing rate (1/ISI) from the
    # detected spikes so the overview's firing-rate panel is populated on the raw path.
    analysis_freqs = np.full(len(analysis_spikes), np.nan)
    if len(analysis_spikes) > 1:
        analysis_freqs[1:] = 1.0 / np.diff(analysis_spikes)

if len(analysis_spikes) == 0:
    st.warning(
        "No spikes were available for analysis with the current input and threshold. "
        "Try lowering the threshold multiplier or checking the uploaded trace."
    )
    st.stop()

# ── Burst detection dispatch ──────────────────────────────────────────────────
spike_tuple   = tuple(analysis_spikes.tolist())
bursts_mi     = None
bursts_logisi = None
isi_th        = None
void_param    = None
logisi_fb     = None
hist_data     = None
hamming_pct   = None
agreement     = None
dual_thresh   = False

with st.spinner("Detecting bursts..."):
    if show_mi:
        bursts_mi = _cached_mi(
            spike_tuple,
            max_beg_isi / 1000, max_end_isi / 1000,
            min_ibi_ms / 1000,  min_dur_ms  / 1000,
            min_spk,
        )
    if show_logisi:
        bursts_logisi, isi_th, void_param, logisi_fb, hist_data = \
            _cached_logisi(spike_tuple, min_spk, void_thresh)
        if not logisi_fb and isi_th is not None and isi_th > 100.0:
            dual_thresh = True
    if burst_method == "Both — compare methods":
        hamming_pct, agreement = compare_methods(bursts_mi, bursts_logisi, rec_dur)

# Primary bursts for display in overview/ISI/burst tabs
if burst_method == "Max Interval (Cotterill et al. 2016)":
    bursts   = bursts_mi
    isi_disp = max_beg_isi
    sec_disp = None
elif burst_method == "logISI Adaptive (Pasquale et al. 2010)":
    bursts   = bursts_logisi
    isi_disp = isi_th
    sec_disp = isi_th if dual_thresh else None
else:
    bursts   = bursts_mi
    isi_disp = max_beg_isi
    sec_disp = None

if has_raw_trace:
    with st.spinner("Extracting waveforms..."):
        waveforms, troughs, peaks, p2p, valid_times, t_axis = extract_waveforms(
            raw_t, raw_filt, analysis_spikes, fs, pre_ms, post_ms
        )
elif ne_mc_waveforms is not None:
    # Pre-extracted per-spike waveform snippets bundled in the export — no raw
    # trace to slice, so use them directly. The file carries no per-sample timing
    # for these, hence the fixed known recording sample rate and -1 to +2 ms
    # acquisition window confirmed for this export format.
    waveforms = ne_mc_waveforms
    valid_times = analysis_spikes
    troughs, peaks, p2p = waveform_amplitude_stats(waveforms)
    t_axis = np.arange(waveforms.shape[1]) / NE_MULTICHANNEL_WAVEFORM_FS * 1000 - 1.0
else:
    waveforms, troughs, peaks, p2p, valid_times, t_axis = (
        np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    )

widths_ms = compute_spike_widths(waveforms, t_axis) if len(waveforms) else np.array([])
isi_pre, isi_post = compute_isi_arrays(valid_times)
snr_arr = np.abs(troughs) / noise_floor if (has_raw_trace and len(troughs)) else np.array([])
has_waveforms = has_raw_trace or (ne_mc_waveforms is not None)
burst_stats = compute_burst_amplitude_stats(bursts, valid_times, p2p, widths_ms) if bursts else []
in_burst_mask = (burst_membership_mask(burst_stats, len(valid_times))
                  if burst_stats else np.zeros(len(valid_times), dtype=bool))
decr_positions, decr_amps, decr_burst_ids = (
    compute_intraburst_decrement(burst_stats, p2p) if burst_stats
    else (np.array([]), np.array([]), np.array([]))
)

# ── Method comparison banner ──────────────────────────────────────────────────
if burst_method == "Both — compare methods" and hamming_pct is not None:
    mi_n, log_n = len(bursts_mi), len(bursts_logisi)
    agree_plain = (
        "Both methods agree — burst calls are robust."
        if hamming_pct < 5 else
        "Results are broadly consistent across methods."
        if hamming_pct <= 10 else
        "Methods disagree substantially — review parameters or ISI structure."
    )
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("MI Bursts",     str(mi_n))
    hc2.metric("logISI Bursts", str(log_n))
    hc3.metric("Hamming Dist.", f"{hamming_pct:.1f}%")
    hc4.metric("Agreement",     _strip_emoji(agreement).split()[0])
    st.info(
        f"The two methods agreed on **{100 - hamming_pct:.1f}%** of the recording. "
        f"{agree_plain}"
    )

# ── Top-level summary metrics ─────────────────────────────────────────────────
st.markdown("## Summary")
n_spikes  = len(analysis_spikes)
mean_fr   = n_spikes / rec_dur if rec_dur > 0 else float('nan')
n_bursts  = len(bursts)
pct_burst = 100 * sum(b['n_spikes'] for b in bursts) / max(n_spikes, 1)
mean_p2p  = float(p2p.mean()) if len(p2p) > 0 else 0.0
snr       = abs(float(troughs.mean())) / noise_floor if (has_raw_trace and len(troughs) > 0) else 0.0

base_metrics = [
    ("Total Spikes",     str(n_spikes)),
    ("Recording (s)",    f"{rec_dur:.2f}"),
    ("Mean Firing Rate", f"{mean_fr:.2f} Hz" if not np.isnan(mean_fr) else "—"),
    ("Bursts Detected",  str(n_bursts)),
    ("Spikes in Bursts", f"{pct_burst:.1f}%"),
    ("Mean P2P Amp",     f"{mean_p2p:.1f} µV"),
]
if show_logisi:
    base_metrics += [
        ("ISIth (ms)",    f"{isi_th:.1f}" if isi_th is not None else "—"),
        ("Void parameter", f"{void_param:.3f}" if void_param is not None else "—"),
    ]
if burst_method == "Both — compare methods":
    base_metrics.append(
        ("Method agreement", f"{hamming_pct:.1f}%" if hamming_pct is not None else "—")
    )

# Wrap metrics into rows of at most six so cards stay legible when the burst
# method adds logISI / comparison metrics (up to nine total) instead of cramming
# them all into one row of equal-width columns.
_PER_ROW = 6
for _start in range(0, len(base_metrics), _PER_ROW):
    _chunk = base_metrics[_start:_start + _PER_ROW]
    for col, (label, value) in zip(st.columns(_PER_ROW), _chunk):
        col.metric(label, value)

if not has_raw_trace:
    st.success(
        f"Using **{n_spikes} pre-sorted spike timestamps** from channel "
        f"**{ne_mc_channel['channel_id']}** — **{ne_mc_channel['label']}** "
        "(standalone NeuroExplorer multichannel waveform export)."
        + (" Waveform snippets were bundled with this channel." if ne_mc_waveforms is not None else "")
    )
elif ne_spike_times is not None:
    st.success(
        f"Using **{len(ne_spike_times)} spike timestamps** from NeuroExplorer export. "
        f"Detection threshold: **{threshold:.2f} µV** ({thr_mult}× σ) — "
        f"noise floor: **{noise_floor:.3f} µV** — SNR: **{snr:.1f}×**"
    )
else:
    st.info(
        f"Spikes detected from raw signal. "
        f"Threshold: **{threshold:.2f} µV** ({thr_mult}× σ) — "
        f"noise floor: **{noise_floor:.3f} µV** — SNR: **{snr:.1f}×**"
    )

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
# Tabs that need a raw trace or extracted waveforms are only included when this
# channel actually has that data (e.g. a multi-channel NeuroExplorer channel with
# no bundled waveform samples has neither) — omitted rather than shown empty.
_tab_specs = [("tab1", "Overview")]
if has_raw_trace:
    _tab_specs.append(("tab_raw", "Raw Trace"))
_tab_specs.append(("tab2", "ISI Analysis"))
if has_waveforms:
    _tab_specs += [("tab3", "Amplitude"), ("tab_wave", "Waveform Metrics")]
_tab_specs += [
    ("tab4", "Bursts"), ("tab_dyn", "Burst Amplitude Dynamics"),
    ("tab5", "logISI Histogram"), ("tab6", "Data Table"),
]
_tabs = dict(zip((k for k, _ in _tab_specs), st.tabs([label for _, label in _tab_specs])))

with _tabs["tab1"]:
    st.markdown("#### Time Window")
    full_overview = st.checkbox("Show full recording", value=True)
    overview_range = None
    if not full_overview:
        rec_start = float(raw_t[0]) if has_raw_trace else float(analysis_spikes[0])
        rec_end = float(raw_t[-1]) if has_raw_trace else float(analysis_spikes[-1])
        default_width = min(10.0, rec_end - rec_start)
        overview_range = st.slider(
            "Visible time range (s)",
            min_value=rec_start,
            max_value=rec_end,
            value=(rec_start, rec_start + default_width),
            step=0.1,
        )
    fig_ov, panels_ov = plot_overview(analysis_spikes, bursts, analysis_freqs, overview_range)
    _render_figure(fig_ov, "overview.png", panels_ov, "overview")

if has_raw_trace:
    with _tabs["tab_raw"]:
        st.markdown("#### Continuous Voltage Trace")
        view_mode = st.radio(
            "View",
            options=["Filtered (bandpass)", "Raw (unfiltered)"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if view_mode == "Raw (unfiltered)":
            fig_raw = plot_raw_trace(raw_t, raw_v, "Raw", C_MUTED)
            fname = "raw_trace.png"
        else:
            fig_raw = plot_raw_trace(raw_t, raw_filt, "Filtered", C_SPIKE)
            fname = "filtered_trace.png"
        _render_figure(fig_raw, fname, combined=False)

with _tabs["tab2"]:
    if len(analysis_spikes) > 1:
        fig_isi, panels_isi = plot_isi(analysis_spikes, isi_disp, sec_disp)
        _render_figure(fig_isi, "isi_analysis.png", panels_isi, "isi_analysis")
        isis_all = np.diff(analysis_spikes) * 1000
        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("Mean ISI", f"{isis_all.mean():.1f} ms")
        ic2.metric("Intra-burst ISI",
                   f"{isis_all[isis_all <= isi_disp].mean():.1f} ms" if any(isis_all <= isi_disp) else "—")
        ic3.metric("Inter-burst ISI",
                   f"{isis_all[isis_all > isi_disp].mean():.0f} ms"  if any(isis_all > isi_disp)  else "—")

if has_waveforms:
    with _tabs["tab3"]:
        if not has_raw_trace:
            st.caption("No detection threshold applies — spikes are pre-sorted "
                       "from NeuroExplorer, not threshold-detected.")
        if len(waveforms) > 0:
            fig_amp, panels_amp = plot_amplitude(waveforms, troughs, peaks, p2p,
                                      valid_times, t_axis, noise_floor, threshold)
            _render_figure(fig_amp, "amplitude.png", panels_amp, "amplitude")
            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("Mean Trough", f"{troughs.mean():.2f} µV")
            ac2.metric("Mean P2P",    f"{p2p.mean():.2f} µV")
            ac3.metric("P2P Std Dev", f"{p2p.std():.2f} µV")
            ac4.metric("SNR",         f"{snr:.1f}×" if has_raw_trace else "—")

            if in_burst_mask.any() and (~in_burst_mask).any():
                fig_ib = plot_amplitude_burst_membership(p2p, in_burst_mask)
                _render_figure(fig_ib, "amplitude_burst_membership.png", combined=False)

    with _tabs["tab_wave"]:
        if not has_raw_trace:
            st.caption("Widths assume the recording's known 25 kHz sample rate for "
                       "these waveform snippets; SNR isn't applicable (no detection threshold).")
        if len(widths_ms) > 0:
            fig_wm, panels_wm = plot_waveform_metrics(widths_ms, p2p, snr_arr)
            _render_figure(fig_wm, "waveform_metrics.png", panels_wm, "waveform_metrics")
        else:
            st.info("No waveforms available to compute width/SNR metrics.")

with _tabs["tab4"]:
    if bursts:
        fig_b, panels_b = plot_bursts(bursts, p2p, in_burst_mask)
        if fig_b:
            _render_figure(fig_b, "bursts.png", panels_b, "bursts")
        durs      = [b['duration'] for b in bursts]
        ns        = [b['n_spikes'] for b in bursts]
        ibis_list = [(bursts[i+1]['start'] - bursts[i]['end']) * 1000
                     for i in range(len(bursts) - 1)]
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Bursts",            str(len(bursts)))
        bc2.metric("Mean Duration",     f"{np.mean(durs):.1f} ms")
        bc3.metric("Mean Spikes/Burst", f"{np.mean(ns):.1f}")
        bc4.metric("Mean IBI",          f"{np.mean(ibis_list):.0f} ms" if ibis_list else "—")
        st.dataframe(pd.DataFrame([{
            'Burst':             f"B{k+1}",
            'Start (s)':         round(b['start'],    4),
            'End (s)':           round(b['end'],      4),
            'Duration (ms)':     round(b['duration'], 1),
            'Spikes':            b['n_spikes'],
            'Mean Amp (µV)':     round(b['mean_amp'], 2) if not np.isnan(b['mean_amp']) else None,
            'Max Amp (µV)':      round(b['max_amp'],  2) if not np.isnan(b['max_amp'])  else None,
            'SD Amp (µV)':       round(b['sd_amp'],   2) if not np.isnan(b['sd_amp'])   else None,
            'CV Amp':            round(b['cv_amp'],   3) if not np.isnan(b['cv_amp'])   else None,
            'Attenuation Index': round(b['attenuation_index'], 3) if not np.isnan(b['attenuation_index']) else None,
        } for k, b in enumerate(burst_stats)]), use_container_width=True, hide_index=True)

        st.markdown("#### Burst-Level Correlations")
        fig_bc, panels_bc = plot_burst_correlations(burst_stats)
        if fig_bc:
            _render_figure(fig_bc, "burst_correlations.png", panels_bc, "burst_correlations")
    else:
        st.warning("No bursts were detected with the current parameters.")

with _tabs["tab_dyn"]:
    if len(decr_positions) > 0 or (~np.isnan(isi_pre)).any():
        fig_dyn, panels_dyn = plot_burst_amplitude_dynamics(decr_positions, decr_amps, valid_times, p2p, isi_pre, isi_post)
        _render_figure(fig_dyn, "burst_amplitude_dynamics.png", panels_dyn, "burst_amplitude_dynamics")
    else:
        st.info("Not enough multi-spike bursts / ISIs to compute amplitude dynamics.")

with _tabs["tab5"]:
    st.markdown("#### logISI Histogram — Pasquale et al. (2010)")
    if burst_method == "Max Interval (Cotterill et al. 2016)":
        st.info(
            "This histogram is shown for inspection only. "
            "Select **logISI Adaptive** or **Both** to use it for burst detection."
        )
        _b, _ith, _vp, _fb, _hd = _cached_logisi(spike_tuple, min_spk, void_thresh)
        fig_logi = plot_logisi_histogram(_hd, _ith, _vp, _fb)
    else:
        fig_logi = plot_logisi_histogram(hist_data, isi_th, void_param, logisi_fb)

    _render_figure(fig_logi, "logisi_histogram.png", combined=False)

    if burst_method == "Both — compare methods":
        st.markdown("#### Burst Epoch Comparison Raster")
        fig_cmp, panels_cmp = plot_comparison_raster(
            analysis_spikes, bursts_mi, bursts_logisi, hamming_pct, agreement
        )
        _render_figure(fig_cmp, "comparison_raster.png", panels_cmp, "comparison_raster")

with _tabs["tab6"]:
    if len(valid_times) > 0:
        df = build_summary_df(analysis_spikes, bursts, troughs, p2p, valid_times, noise_floor, in_burst_mask)
        if not has_raw_trace:
            df.insert(0, 'Electrode ID', ne_mc_channel['channel_id'])
            df.insert(1, 'Source Label', ne_mc_channel['label'])
        if len(widths_ms) == len(df):
            df['Spike Width (ms)']   = np.round(widths_ms, 3)
            df['Preceding ISI (ms)'] = np.round(isi_pre, 2)
            df['Following ISI (ms)'] = np.round(isi_post, 2)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={
                         'SNR (×σ)':           st.column_config.NumberColumn(format="%.1f"),
                         'Trough (µV)':         st.column_config.NumberColumn(format="%.2f"),
                         'Peak-to-Peak (µV)':   st.column_config.NumberColumn(format="%.2f"),
                     })
        st.download_button("Download CSV",
                           df.to_csv(index=False).encode(),
                           "spike_data.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# Methods export
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Export Methods Text")
st.markdown("Generate a formatted paragraph suitable for a methods section.")

if st.button("Generate Methods Text"):
    if burst_method == "Max Interval (Cotterill et al. 2016)":
        method_name  = "Max Interval"
        citation     = "(Cotterill et al. 2016, J Neurophysiol)"
        method_frag  = (
            f"The following parameters were used: maximum beginning ISI = {max_beg_isi} ms, "
            f"maximum end ISI = {max_end_isi} ms, minimum interburst interval = {min_ibi_ms} ms, "
            f"minimum burst duration = {min_dur_ms} ms, minimum spikes per burst = {min_spk}."
        )
        compare_frag = ""
    elif burst_method == "logISI Adaptive (Pasquale et al. 2010)":
        method_name  = "logISI adaptive"
        citation     = "(Pasquale et al. 2010, J Comput Neurosci)"
        _ith  = f"{isi_th:.1f}" if isi_th is not None else "100.0"
        _vp   = f"{void_param:.2f}" if void_param is not None else "N/A"
        _comp = "exceeded" if (isi_th is not None and isi_th > 100) else "did not exceed"
        _mode = "dual-threshold boundary extension" if dual_thresh else "single threshold"
        fb_note = " The method fell back to the fixed 100 ms criterion." if logisi_fb else ""
        method_frag  = (
            f"The ISI threshold was automatically determined from the logarithmic ISI "
            f"histogram as ISIth = {_ith} ms (void parameter = {_vp}), which {_comp} 100 ms, "
            f"so {_mode} was applied.{fb_note}"
        )
        compare_frag = ""
    else:
        method_name  = "Max Interval (primary) and logISI adaptive (comparison)"
        citation     = "(Cotterill et al. 2016, J Neurophysiol; Pasquale et al. 2010, J Comput Neurosci)"
        _ith  = f"{isi_th:.1f}" if isi_th is not None else "100.0"
        _vp   = f"{void_param:.2f}" if void_param is not None else "N/A"
        agree_word = (
            "strong"   if (hamming_pct is not None and hamming_pct < 5)  else
            "moderate" if (hamming_pct is not None and hamming_pct <= 10) else
            "poor"
        )
        method_frag  = (
            f"For the Max Interval method: maximum beginning ISI = {max_beg_isi} ms, "
            f"maximum end ISI = {max_end_isi} ms, minimum interburst interval = {min_ibi_ms} ms, "
            f"minimum burst duration = {min_dur_ms} ms, minimum spikes per burst = {min_spk}. "
            f"For the logISI method: ISIth = {_ith} ms (void parameter = {_vp})."
        )
        _hd = f"{hamming_pct:.1f}" if hamming_pct is not None else "N/A"
        compare_frag = (
            f" Both methods were applied and showed {agree_word} agreement "
            f"(Hamming distance = {_hd}%; Cotterill et al. 2016)."
        )

    if has_raw_trace:
        detection_frag = (
            f"Spikes were detected as negative threshold crossings exceeding {thr_mult}× "
            f"the estimated noise floor, computed as the median absolute deviation of the "
            f"bandpass-filtered signal ({int(bp_low)}–{int(bp_high)} Hz, 4th-order Butterworth; "
            f"Quiroga et al. 2004), with a 1 ms refractory period. "
        )
        amplitude_frag = (
            f"Spike amplitude was quantified as peak-to-peak voltage of the filtered waveform "
            f"within a {pre_ms} ms–{post_ms} ms window around each threshold crossing "
            f"(Obien et al. 2015)."
        )
    else:
        detection_frag = (
            f"Spikes were taken directly from pre-sorted timestamps for electrode "
            f"{ne_mc_channel['channel_id']} (source label: {ne_mc_channel['label']}), "
            f"exported from NeuroExplorer; recording duration was {rec_dur:.3f} s. "
        )
        amplitude_frag = (
            f"Bundled 76-sample waveforms were acquired at 25 kHz from -1.0 to +2.0 ms; "
            f"values declared by the user as {waveform_unit} were normalized to µV, and "
            "spike amplitude was quantified as peak-to-peak voltage (Obien et al. 2015)."
            if ne_mc_waveforms is not None else ""
        )

    methods_text = (
        f"{detection_frag}"
        f"Bursts were identified using the {method_name} method {citation}. "
        f"{method_frag}{compare_frag} "
        f"{amplitude_frag}"
    )
    st.text_area("Methods section paragraph", methods_text, height=200)

st.markdown("---")
st.caption(
    "MEA Spike Analyser — "
    "Cotterill et al. (2016) J Neurophysiol · "
    "Pasquale et al. (2010) J Comput Neurosci"
)
