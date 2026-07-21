from datetime import datetime, timezone

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from parsers import (  # noqa: E402
    MAX_EDF_UPLOAD_BYTES,
    edf_signal_labels,
    format_bytes,
    is_ne_multichannel_content,
    parse_edf_content,
    parse_ne_multichannel_content,
    parse_neuroexplorer_content,
    parse_raw_content,
    read_edf_upload_bytes,
    stitch_segments,
    validate_edf_upload_size,
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
    plot_spike_waveforms,
    plot_waveform_metrics,
    render_panel_downloads,
)
from processing import (  # noqa: E402
    METHODS_REVIEW_WARNING,
    analysis_metadata_to_json,
    bandpass_filter,
    build_analysis_metadata,
    build_burst_summary_df,
    build_summary_df,
    burst_membership_mask,
    compare_methods,
    compute_burst_amplitude_stats,
    compute_intraburst_decrement,
    compute_isi_arrays,
    compute_spike_widths,
    detect_spikes,
    extract_waveforms,
    infer_sampling_rate,
    logisi_method,
    max_interval_method,
    validate_spike_timestamps,
    waveform_amplitude_stats,
)

# Sampling rate for NeuroExplorer multi-channel waveform snippets — this format
# carries no per-file timing metadata for Spike_value_1..N samples; 25 kHz is the
# recording system's known rate, confirmed for this lab's exports.
NE_MULTICHANNEL_WAVEFORM_FS = 25_000
EDF_MEMORY_ERROR_MESSAGE = (
    "The EDF file could not be processed with the available memory. EDF parsing, "
    "voltage calibration, timestamp generation, and filtering require additional "
    "in-memory arrays. Use a shorter recording or smaller channel subset and try again."
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpikeLab",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"],
    [data-testid="stSidebarContent"] {
        padding-top: 2rem;
    }
    [data-testid="stMetricValue"] {
        font-size: clamp(1.4rem, 2vw, 2rem);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached parsing wrappers ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def parse_raw_file(content: bytes):
    return parse_raw_content(content)

@st.cache_data(show_spinner=False)
def parse_neuroexplorer_file(content: bytes):
    return parse_neuroexplorer_content(content)

# Keyed on file identity — the leading-underscore `_content` is not hashed, so
# switching the selected channel never re-runs the whole-file positional parse.
@st.cache_data(show_spinner=False)
def parse_ne_multichannel_file(file_key: str, _content: bytes):
    return parse_ne_multichannel_content(_content)

# Read an upload's bytes once per file (leading-underscore `_uploaded` is not
# hashed). EDF callers validate the size before this cached materialisation.
@st.cache_data(show_spinner=False)
def _read_upload_bytes(file_key: str, _uploaded, max_size_bytes=None):
    if max_size_bytes is not None:
        return read_edf_upload_bytes(_uploaded, max_size_bytes)
    return _uploaded.getvalue()

# Keyed on (file identity, channel) — the leading-underscore `_content` is not
# hashed, so switching channels or nudging a slider does not re-hash the EDF.
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


def _stop_for_edf_memory_error():
    st.error(EDF_MEMORY_ERROR_MESSAGE)
    st.stop()


def _select_subview(label, options, key, default):
    if not options:
        return None
    if len(options) == 1:
        return options[0]

    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
    if key in st.session_state:
        return st.segmented_control(
            label,
            options,
            key=key,
            label_visibility="collapsed",
        )
    return st.segmented_control(
        label,
        options,
        default=default if default in options else options[0],
        key=key,
        label_visibility="collapsed",
    )


def _render_figure(
    fig,
    filename,
    panels=None,
    base_filename=None,
    display_notes=None,
    title=None,
    related_figures=None,
):
    related_figures = related_figures or []
    download_key = base_filename or filename.removesuffix(".png")
    title_col, download_col = st.columns([5, 1], vertical_alignment="center")
    if title:
        title_col.markdown(f"#### {title}")

    if (panels and base_filename) or related_figures:
        with download_col.popover(
            "Download ▾",
            key=f"download_menu_{download_key}",
            width="stretch",
        ):
            st.download_button(
                "Download combined PNG",
                fig_to_bytes(fig),
                filename,
                "image/png",
                key=f"dl_{download_key}_combined",
                width="stretch",
            )
            render_panel_downloads(fig, panels, base_filename)
            for index, (related_title, related_fig, related_filename) in enumerate(related_figures):
                st.download_button(
                    f"Download {related_title.lower()} PNG",
                    fig_to_bytes(related_fig),
                    related_filename,
                    "image/png",
                    key=f"dl_{download_key}_related_{index}",
                    width="stretch",
                )
    else:
        download_col.download_button(
            "Download PNG",
            fig_to_bytes(fig),
            filename,
            "image/png",
            key=f"dl_{download_key}_single",
            width="stretch",
        )

    st.pyplot(fig, width="stretch")
    for note in display_notes or []:
        st.caption(note)
    plt.close(fig)
    for related_title, related_fig, _ in related_figures:
        st.markdown(f"#### {related_title}")
        st.pyplot(related_fig, width="stretch")
        plt.close(related_fig)

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.title("SpikeLab")
st.markdown("### Offline MEA spike and burst analysis workbench")
st.caption(
    "Spike detection, burst analysis, and waveform quantification for "
    "Multi-Electrode Array recordings."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Analysis settings")

    st.markdown("**Spike detection**")
    thr_mult = st.slider(
        "Threshold multiplier",
        3.0,
        10.0,
        5.0,
        0.5,
        help="Multiple of the MAD-derived noise estimate. The standard is 5×; "
             "higher values are more conservative and detect fewer spikes.",
    )
    bp_low   = st.number_input("Bandpass low cutoff (Hz)",  100, 1000, 300, 50)
    bp_high  = st.number_input("Bandpass high cutoff (Hz)", 1000, 6000, 3000, 100)

    st.markdown("---")
    st.markdown("**Burst detection**")
    burst_method = st.radio(
        "Method",
        options=[
            "Max Interval",
            "Adaptive logISI",
            "Compare both",
        ],
        label_visibility="collapsed",
    )
    st.caption(
        "Max Interval: Cotterill et al. (2016), J Neurophysiol · "
        "Adaptive logISI: Pasquale et al. (2010), J Comput Neurosci"
    )

    show_mi     = burst_method != "Adaptive logISI"
    show_logisi = burst_method != "Max Interval"

    if show_mi:
        st.markdown("**Max Interval parameters**")
        max_beg_isi = st.slider("Maximum beginning ISI (ms)",  50,  500, 170, 10)
        max_end_isi = st.slider("Maximum ending ISI (ms)",     100, 1000, 300, 10)
        min_ibi_ms  = st.slider("Minimum interburst interval (ms)", 50, 1000, 200, 10)
        min_dur_ms  = st.slider("Minimum burst duration (ms)",  5, 200, 10, 5)
    else:
        max_beg_isi = max_end_isi = min_ibi_ms = min_dur_ms = None

    if show_mi and show_logisi:
        st.markdown("---")

    if show_logisi:
        st.markdown("**logISI parameters**")
        void_thresh = st.slider("Void parameter threshold", 0.0, 1.0, 0.7, 0.05)
    else:
        void_thresh = 0.7

    min_spk = st.slider("Minimum spikes per burst", 2, 10, 3, 1)

    st.markdown("---")
    st.markdown("**Waveform settings**")
    pre_ms  = st.slider("Pre-spike window (ms)",  0.5, 3.0, 1.0, 0.25)
    post_ms = st.slider("Post-spike window (ms)", 1.0, 5.0, 2.0, 0.25)

# ── File upload ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    raw_files = st.file_uploader(
        "Recording or waveform export",
        type=['txt', 'csv', 'edf'],
        accept_multiple_files=True,
        help="Upload continuous TXT/CSV or EDF data, or one standalone NeuroExplorer "
             "multichannel waveform export.",
    )
    st.caption(
        "TXT/CSV continuous trace · EDF recording · NeuroExplorer waveform export"
    )
    st.caption(f"EDF upload limit: {format_bytes(MAX_EDF_UPLOAD_BYTES)}.")
with col2:
    ne_file = st.file_uploader(
        "Optional spike-time overlay",
        type=['txt'],
        help="Optional NeuroExplorer Instantaneous Parameters TXT file for a continuous recording.",
    )
    st.caption("NeuroExplorer TXT timestamps used with a continuous recording")

if not raw_files:
    st.info(
        "Upload a continuous recording or standalone NeuroExplorer waveform export. "
        "Optionally add pre-detected NeuroExplorer spike timestamps."
    )
    with st.expander("Supported input formats", expanded=False):
        st.markdown(
            f"""
- **TXT/CSV continuous trace:** Two columns containing time in seconds and voltage in µV.
  Multiple sequential segments can be uploaded together and stitched.
- **EDF recording:** One continuous recording up to {format_bytes(MAX_EDF_UPLOAD_BYTES)}.
  Select one signal channel for in-memory analysis.
- **NeuroExplorer waveform export:** One standalone multichannel TXT export. The selected
  electrode's pre-sorted spikes are analysed without a continuous raw trace.
- **Optional spike-time overlay:** A NeuroExplorer TXT file with an `Instantaneous Parameters`
  section. It replaces detected spike times for a continuous TXT/CSV or EDF recording and
  cannot be combined with the standalone multichannel waveform export.
"""
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
    for channel in ne_mc_channels:
        try:
            channel['spike_times'] = validate_spike_timestamps(channel['spike_times'])
        except ValueError as exc:
            st.error(
                f"Invalid spike timestamps in NeuroExplorer channel "
                f"{channel['channel_id']} ({channel['label']}): {exc}"
            )
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
    try:
        validate_edf_upload_size(edf_file.size)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.caption(
        f"EDF file size: {format_bytes(edf_file.size)}. "
        "The selected channel will be processed in memory."
    )
    file_key = getattr(edf_file, 'file_id', None) or f"{edf_file.name}:{getattr(edf_file, 'size', 0)}"
    try:
        edf_bytes = _read_upload_bytes(file_key, edf_file, MAX_EDF_UPLOAD_BYTES)
        edf_channels = edf_signal_labels(edf_bytes)
    except MemoryError:
        _stop_for_edf_memory_error()
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
        except MemoryError:
            _stop_for_edf_memory_error()
        except ValueError as exc:
            st.error(f"Could not read the selected EDF channel: {exc}")
            st.stop()
    try:
        validation_errors = _segment_validation_errors(raw_t, raw_v, edf_file.name)
    except MemoryError:
        _stop_for_edf_memory_error()
    if validation_errors:
        st.error("The EDF channel could not be analysed:\n\n" + "\n".join(f"- {e}" for e in validation_errors))
        st.stop()
else:
    with st.spinner("Loading signal..."):
        segments = []
        validation_errors = []
        for f in txt_files:
            try:
                t, v = parse_raw_file(f.read())
            except ValueError as exc:
                st.error(f"Could not parse raw trace {f.name}: {exc}")
                st.stop()
            validation_errors.extend(_segment_validation_errors(t, v, f.name))
            segments.append((t, v))
        if validation_errors:
            st.error("The uploaded raw trace could not be analysed:\n\n" + "\n".join(f"- {e}" for e in validation_errors))
            st.stop()
        if len(segments) == 1:
            raw_t, raw_v = segments[0]
        else:
            try:
                raw_t, raw_v, stitch_summary = stitch_segments(segments)
            except ValueError as exc:
                st.error(f"The uploaded raw segments could not be stitched: {exc}")
                st.stop()

ne_spike_times, ne_isis, ne_freqs = None, None, None

if has_raw_trace:
    # ── Resolve sampling rate and filter continuous input ────────────────────
    with st.spinner("Filtering signal..."):
        if edf_files:
            fs = edf_channels[sel][2]
        else:
            try:
                fs = infer_sampling_rate(raw_t)
            except ValueError as exc:
                st.error(f"Could not infer the text/CSV sampling rate: {exc}")
                st.stop()
        try:
            raw_filt = bandpass_filter(raw_v, fs, bp_low, bp_high)
        except MemoryError:
            if not edf_files:
                raise
            _stop_for_edf_memory_error()
        except ValueError as exc:
            st.error(f"Could not filter the uploaded trace: {exc}")
            st.stop()

    if stitch_summary is not None:
        with st.expander(
            f"{len(txt_files)} segments stitched · "
            f"{raw_t[-1] - raw_t[0]:.1f} s recording",
            expanded=False,
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
            } for k, row in enumerate(stitch_summary)]), width="stretch", hide_index=True)

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
        else:
            try:
                ne_spike_times = validate_spike_timestamps(
                    ne_spike_times, recording_range=(raw_t[0], raw_t[-1])
                )
            except ValueError as exc:
                st.error(f"The NeuroExplorer spike-time overlay is invalid: {exc}")
                st.stop()

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
    if burst_method == "Compare both":
        recording_start = float(raw_t[0]) if has_raw_trace else 0.0
        hamming_pct, agreement = compare_methods(
            bursts_mi, bursts_logisi, rec_dur, recording_start=recording_start
        )

# Primary bursts for display in overview/ISI/burst tabs
if burst_method == "Max Interval":
    bursts   = bursts_mi
    isi_disp = max_beg_isi
    sec_disp = None
elif burst_method == "Adaptive logISI":
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
burst_export_method = (
    "logISI Adaptive"
    if burst_method == "Adaptive logISI"
    else "Max Interval"
)
burst_df = build_burst_summary_df(burst_stats, burst_export_method)

source_filenames = [f.name for f in raw_files]
if ne_file is not None:
    source_filenames.append(ne_file.name)

if ne_mc_files:
    source_input_type = "standalone_neuroexplorer"
    selected_channel = {
        'id': ne_mc_channel['channel_id'],
        'label': ne_mc_channel['label'],
    }
    parser_information = {
        'format': 'neuroexplorer_multichannel_waveform',
        'waveforms_available': ne_mc_waveforms is not None,
        'waveform_samples_per_spike': (
            ne_mc_waveforms.shape[1] if ne_mc_waveforms is not None else None
        ),
    }
    stitch_information = None
    recording_start = 0.0
    recording_end = rec_dur
    analysis_sampling_rate = (
        NE_MULTICHANNEL_WAVEFORM_FS if ne_mc_waveforms is not None else None
    )
    spike_source = "standalone_neuroexplorer"
    selected_waveform_unit = waveform_unit
elif edf_files:
    source_input_type = "edf"
    selected_channel = {
        'id': channel_index,
        'label': edf_channels[sel][1],
    }
    parser_information = {
        'format': 'edf',
        'sample_count': edf_channels[sel][3],
        'declared_physical_dimension': edf_channels[sel][4],
        'overlay_filename': ne_file.name if ne_file is not None else None,
    }
    stitch_information = None
    recording_start = raw_t[0]
    recording_end = raw_t[-1]
    analysis_sampling_rate = fs
    spike_source = "neuroexplorer_overlay" if ne_spike_times is not None else "raw_detector"
    selected_waveform_unit = None
else:
    source_input_type = "text_csv"
    selected_channel = None
    parser_information = {
        'format': 'two_column_text_csv',
        'delimiter_detection': 'first_numeric_candidate_row',
        'overlay_filename': ne_file.name if ne_file is not None else None,
    }
    stitch_information = {
        'segment_count': len(txt_files),
        'stitched': stitch_summary is not None,
        'segments': (
            [{
                'order': order + 1,
                'filename': txt_files[row['segment_index']].name,
                'start_s': row['first_t'],
                'end_s': row['last_t'],
                'samples': row['n_samples'],
                'overlap_trimmed_s': row['overlap_s'],
                'overlap_trimmed_samples': row['overlap_samples'],
                'samples_kept': row['n_kept'],
            } for order, row in enumerate(stitch_summary)]
            if stitch_summary is not None else None
        ),
    }
    recording_start = raw_t[0]
    recording_end = raw_t[-1]
    analysis_sampling_rate = fs
    spike_source = "neuroexplorer_overlay" if ne_spike_times is not None else "raw_detector"
    selected_waveform_unit = None

analysis_parameters = {
    'minimum_spikes_per_burst': min_spk,
}
if has_raw_trace:
    analysis_parameters['spike_detection'] = {
        'threshold_multiplier': thr_mult,
        'bandpass_low_hz': bp_low,
        'bandpass_high_hz': bp_high,
        'filter_order': 4,
        'refractory_period_ms': 1.0,
    }
    analysis_parameters['waveform_extraction'] = {
        'pre_spike_ms': pre_ms,
        'post_spike_ms': post_ms,
    }
elif ne_mc_waveforms is not None:
    analysis_parameters['supplied_waveforms'] = {
        'sampling_rate_hz': NE_MULTICHANNEL_WAVEFORM_FS,
        'start_ms': -1.0,
        'end_ms': 2.0,
    }
if show_mi:
    analysis_parameters['max_interval'] = {
        'maximum_beginning_isi_ms': max_beg_isi,
        'maximum_ending_isi_ms': max_end_isi,
        'minimum_interburst_interval_ms': min_ibi_ms,
        'minimum_burst_duration_ms': min_dur_ms,
    }
if show_logisi:
    analysis_parameters['logisi'] = {
        'void_threshold': void_thresh,
    }

selected_burst_method = {
    "Max Interval": "max_interval",
    "Adaptive logISI": "logisi",
    "Compare both": "both",
}[burst_method]
logisi_results = (
    {
        'threshold_ms': isi_th,
        'void_parameter': void_param,
        'fallback': logisi_fb,
        'dual_threshold': dual_thresh,
    }
    if show_logisi else None
)
method_comparison = (
    {
        'hamming_disagreement_percent': hamming_pct,
        'bin_size_ms': 50.0,
        'agreement_label': _strip_emoji(agreement) if agreement is not None else None,
    }
    if burst_method == "Compare both" else None
)
analysis_metadata = build_analysis_metadata(
    analysis_timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    source_filenames=source_filenames,
    source_input_type=source_input_type,
    selected_channel=selected_channel,
    parser_information=parser_information,
    stitch_information=stitch_information,
    recording_start=recording_start,
    recording_end=recording_end,
    recording_duration=rec_dur,
    sampling_rate=analysis_sampling_rate,
    spike_source=spike_source,
    spike_count=len(analysis_spikes),
    valid_waveform_count=len(valid_times),
    burst_count=len(bursts),
    selected_burst_method=selected_burst_method,
    analysis_parameters=analysis_parameters,
    logisi_results=logisi_results,
    method_comparison=method_comparison,
    waveform_unit=selected_waveform_unit,
)
analysis_metadata_json = analysis_metadata_to_json(analysis_metadata)

in_burst_mask = (burst_membership_mask(burst_stats, len(valid_times))
                  if burst_stats else np.zeros(len(valid_times), dtype=bool))
decr_positions, decr_amps, decr_burst_ids = (
    compute_intraburst_decrement(burst_stats, p2p) if burst_stats
    else (np.array([]), np.array([]), np.array([]))
)

# ── Method comparison banner ──────────────────────────────────────────────────
if burst_method == "Compare both" and hamming_pct is not None:
    mi_n, log_n = len(bursts_mi), len(bursts_logisi)
    occupancy_agreement = (
        "High agreement on burst occupancy."
        if hamming_pct < 5 else
        "Moderate agreement on burst occupancy."
        if hamming_pct <= 10 else
        "Low agreement on burst occupancy; review parameters or ISI structure."
    )
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("MI Bursts",     str(mi_n))
    hc2.metric("logISI Bursts", str(log_n))
    hc3.metric("Hamming Dist.", f"{hamming_pct:.1f}%")
    hc4.metric("Occupancy agreement", _strip_emoji(agreement).split()[0])
    st.info(
        f"Burst occupancy matched in **{100 - hamming_pct:.1f}%** of 50 ms bins. "
        f"{occupancy_agreement}"
    )

# ── Top-level summary metrics ─────────────────────────────────────────────────
st.markdown("## Analysis summary")
n_spikes  = len(analysis_spikes)
mean_fr   = n_spikes / rec_dur if rec_dur > 0 else float('nan')
n_bursts  = len(bursts)
pct_burst = 100 * sum(b['n_spikes'] for b in bursts) / max(n_spikes, 1)
mean_p2p  = float(p2p.mean()) if len(p2p) > 0 else 0.0
snr       = abs(float(troughs.mean())) / noise_floor if (has_raw_trace and len(troughs) > 0) else 0.0

base_metrics = [
    ("Total spikes",          str(n_spikes)),
    ("Recording duration",    f"{rec_dur:.2f} s"),
    ("Mean firing rate",      f"{mean_fr:.2f} Hz" if not np.isnan(mean_fr) else "—"),
    ("Bursts detected",       str(n_bursts)),
    ("Spikes in bursts",      f"{pct_burst:.1f}%"),
    ("Mean P2P amplitude",    f"{mean_p2p:.1f} µV"),
]
if show_logisi:
    base_metrics += [
        ("ISIth (ms)",    f"{isi_th:.1f}" if isi_th is not None else "—"),
        ("Void parameter", f"{void_param:.3f}" if void_param is not None else "—"),
    ]
if burst_method == "Compare both":
    base_metrics.append(
        ("Hamming distance", f"{hamming_pct:.1f}%" if hamming_pct is not None else "—")
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
    st.caption(
        f"Standalone NeuroExplorer export · {n_spikes} pre-sorted spike timestamps · "
        f"Channel {ne_mc_channel['channel_id']} ({ne_mc_channel['label']})"
        + (" · Waveform snippets included" if ne_mc_waveforms is not None else "")
    )
elif ne_spike_times is not None:
    st.caption(
        f"NeuroExplorer spike-time overlay · {len(ne_spike_times)} timestamps · "
        f"Raw-signal threshold {threshold:.2f} µV ({thr_mult}× noise) · "
        f"Noise {noise_floor:.3f} µV · SNR {snr:.1f}×"
    )
else:
    st.caption(
        f"Detected from raw signal · Threshold {threshold:.2f} µV "
        f"({thr_mult}× noise) · Noise {noise_floor:.3f} µV · SNR {snr:.1f}×"
    )

st.markdown("---")

# ── Grouped analysis navigation ───────────────────────────────────────────────
waveform_view_available = len(waveforms) > 0
signal_views = []
if has_raw_trace:
    signal_views.append("Raw trace")
if waveform_view_available:
    signal_views.append("Spike waveforms")

spike_views = (
    ["Amplitude", "Waveform metrics", "ISI analysis"]
    if waveform_view_available else
    ["ISI analysis"]
)
burst_views = ["Bursts", "Burst dynamics", "Adaptive logISI"]
data_views = ["Spike data", "Burst data", "Exports & methods"]

spike_df = None
if len(valid_times) > 0:
    spike_df = build_summary_df(
        analysis_spikes,
        bursts,
        troughs,
        p2p,
        valid_times,
        noise_floor,
        in_burst_mask,
    )
    if not has_raw_trace:
        spike_df.insert(0, 'Electrode ID', ne_mc_channel['channel_id'])
        spike_df.insert(1, 'Source Label', ne_mc_channel['label'])
    if len(widths_ms) == len(spike_df):
        spike_df['Spike Width (ms)'] = np.round(widths_ms, 3)
        spike_df['Preceding ISI (ms)'] = np.round(isi_pre, 2)
        spike_df['Following ISI (ms)'] = np.round(isi_post, 2)

tab_overview, tab_signal, tab_spikes, tab_bursts, tab_data = st.tabs([
    "Overview",
    "Signal",
    "Spike analysis",
    "Burst analysis",
    "Data & export",
])

with tab_overview:
    st.subheader("Spike activity overview")
    st.markdown("#### Time window")
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
    _render_figure(
        fig_ov,
        "overview.png",
        panels_ov,
        "overview",
        title="Raster and firing rate",
    )

with tab_signal:
    signal_view = _select_subview(
        "Signal view",
        signal_views,
        "signal_subview",
        "Raw trace" if has_raw_trace else "Spike waveforms",
    )
    if signal_view == "Raw trace":
        st.subheader("Continuous voltage trace")
        view_mode = st.radio(
            "View",
            options=["Filtered (bandpass)", "Raw (unfiltered)"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if view_mode == "Raw (unfiltered)":
            fig_raw, notes_raw = plot_raw_trace(raw_t, raw_v, "Raw", C_MUTED)
            fname = "raw_trace.png"
            raw_figure_title = "Raw voltage trace"
        else:
            fig_raw, notes_raw = plot_raw_trace(raw_t, raw_filt, "Filtered", C_SPIKE)
            fname = "filtered_trace.png"
            raw_figure_title = "Filtered voltage trace"
        _render_figure(
            fig_raw,
            fname,
            display_notes=notes_raw,
            title=raw_figure_title,
        )
    elif signal_view == "Spike waveforms":
        st.subheader("Spike waveform snippets")
        if has_raw_trace:
            st.caption(
                "Waveforms extracted around detected or overlaid spike timestamps from "
                "the bandpass-filtered continuous recording."
            )
        else:
            st.caption(
                "These are the selected electrode's pre-extracted NeuroExplorer snippets, "
                "normalized to µV and plotted on the confirmed 25 kHz time axis. They are "
                "not a continuous raw recording and are not filtered again in this app."
            )
        fig_spike_waveforms, notes_spike_waveforms = plot_spike_waveforms(waveforms, t_axis)
        _render_figure(
            fig_spike_waveforms,
            "spike_waveforms.png",
            display_notes=notes_spike_waveforms,
            title="Representative waveforms",
        )
    else:
        st.info("No continuous signal or waveform snippets are available for this input.")

with tab_spikes:
    spike_view = _select_subview(
        "Spike analysis view",
        spike_views,
        "spike_analysis_subview",
        "Amplitude",
    )
    if spike_view == "Amplitude":
        st.subheader("Spike amplitude")
        if not has_raw_trace:
            st.caption(
                "No detection threshold applies — spikes are pre-sorted "
                "from NeuroExplorer, not threshold-detected."
            )
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("Mean trough", f"{troughs.mean():.2f} µV")
        ac2.metric("Mean P2P", f"{p2p.mean():.2f} µV")
        ac3.metric("P2P standard deviation", f"{p2p.std():.2f} µV")
        ac4.metric("SNR", f"{snr:.1f}×" if has_raw_trace else "—")

        fig_amp, panels_amp, notes_amp = plot_amplitude(
            waveforms, troughs, peaks, p2p, valid_times, t_axis, noise_floor, threshold
        )
        related_amplitude_figures = []
        if in_burst_mask.any() and (~in_burst_mask).any():
            fig_ib = plot_amplitude_burst_membership(p2p, in_burst_mask)
            related_amplitude_figures.append((
                "Burst-membership comparison",
                fig_ib,
                "amplitude_burst_membership.png",
            ))
        _render_figure(
            fig_amp,
            "amplitude.png",
            panels_amp,
            "amplitude",
            display_notes=notes_amp,
            title="Amplitude quantification",
            related_figures=related_amplitude_figures,
        )
    elif spike_view == "Waveform metrics":
        st.subheader("Waveform metrics")
        if not has_raw_trace:
            st.caption(
                "Widths assume the recording's known 25 kHz sample rate for "
                "these waveform snippets; SNR isn't applicable (no detection threshold)."
            )
        if len(widths_ms) > 0:
            fig_wm, panels_wm, notes_wm = plot_waveform_metrics(widths_ms, p2p, snr_arr)
            _render_figure(
                fig_wm,
                "waveform_metrics.png",
                panels_wm,
                "waveform_metrics",
                display_notes=notes_wm,
                title="Width, amplitude, and SNR",
            )
        else:
            st.info("No waveforms available to compute width/SNR metrics.")
    elif spike_view == "ISI analysis":
        st.subheader("Inter-spike interval analysis")
        if len(analysis_spikes) <= 1:
            st.info("At least two spikes are required for ISI analysis.")
        else:
            isis_all = np.diff(analysis_spikes) * 1000
            ic1, ic2, ic3 = st.columns(3)
            ic1.metric("Mean ISI", f"{isis_all.mean():.1f} ms")
            ic2.metric(
                "Intra-burst ISI",
                f"{isis_all[isis_all <= isi_disp].mean():.1f} ms"
                if any(isis_all <= isi_disp) else "—",
            )
            ic3.metric(
                "Inter-burst ISI",
                f"{isis_all[isis_all > isi_disp].mean():.0f} ms"
                if any(isis_all > isi_disp) else "—",
            )
            fig_isi, panels_isi = plot_isi(analysis_spikes, isi_disp, sec_disp)
            _render_figure(
                fig_isi,
                "isi_analysis.png",
                panels_isi,
                "isi_analysis",
                title="ISI distributions",
            )

with tab_bursts:
    burst_view = _select_subview(
        "Burst analysis view",
        burst_views,
        "burst_analysis_subview",
        "Bursts",
    )
    if burst_view == "Bursts":
        st.subheader("Detected bursts")
        if not bursts:
            st.warning("No bursts were detected with the current parameters.")
        else:
            durs = [b['duration'] for b in bursts]
            ns = [b['n_spikes'] for b in bursts]
            ibis_list = [
                (bursts[i + 1]['start'] - bursts[i]['end']) * 1000
                for i in range(len(bursts) - 1)
            ]
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Bursts", str(len(bursts)))
            bc2.metric("Mean Duration", f"{np.mean(durs):.1f} ms")
            bc3.metric("Mean Spikes/Burst", f"{np.mean(ns):.1f}")
            bc4.metric(
                "Mean IBI",
                f"{np.mean(ibis_list):.0f} ms" if ibis_list else "—",
            )

            fig_b, panels_b = plot_bursts(bursts, p2p, in_burst_mask)
            if fig_b:
                _render_figure(
                    fig_b,
                    "bursts.png",
                    panels_b,
                    "bursts",
                    title="Burst summary",
                )

            fig_bc, panels_bc, notes_bc = plot_burst_correlations(burst_stats)
            if fig_bc:
                _render_figure(
                    fig_bc,
                    "burst_correlations.png",
                    panels_bc,
                    "burst_correlations",
                    display_notes=notes_bc,
                    title="Burst-level correlations",
                )
    elif burst_view == "Burst dynamics":
        st.subheader("Burst dynamics")
        if len(decr_positions) > 0 or (~np.isnan(isi_pre)).any():
            fig_dyn, panels_dyn, notes_dyn = plot_burst_amplitude_dynamics(
                decr_positions, decr_amps, valid_times, p2p, isi_pre, isi_post
            )
            _render_figure(
                fig_dyn,
                "burst_amplitude_dynamics.png",
                panels_dyn,
                "burst_amplitude_dynamics",
                display_notes=notes_dyn,
                title="Amplitude dynamics",
            )
        else:
            st.info("Not enough multi-spike bursts / ISIs to compute amplitude dynamics.")
    elif burst_view == "Adaptive logISI":
        st.subheader("Adaptive ISI threshold")
        st.caption("logISI method · Pasquale et al. (2010)")
        if burst_method == "Max Interval":
            st.info(
                "This histogram is shown for inspection only. "
                "Select **Adaptive logISI** or **Compare both** to use it for burst detection."
            )
            _b, _ith, _vp, _fb, _hd = _cached_logisi(spike_tuple, min_spk, void_thresh)
            fig_logi = plot_logisi_histogram(_hd, _ith, _vp, _fb)
        else:
            fig_logi = plot_logisi_histogram(hist_data, isi_th, void_param, logisi_fb)

        _render_figure(
            fig_logi,
            "logisi_histogram.png",
            title="logISI distribution",
        )

        if burst_method == "Compare both":
            fig_cmp, panels_cmp = plot_comparison_raster(
                analysis_spikes, bursts_mi, bursts_logisi, hamming_pct, agreement
            )
            _render_figure(
                fig_cmp,
                "comparison_raster.png",
                panels_cmp,
                "comparison_raster",
                title="Burst epoch comparison",
            )

with tab_data:
    data_view = _select_subview(
        "Data and export view",
        data_views,
        "data_export_subview",
        "Spike data",
    )
    if data_view == "Spike data":
        st.subheader("Spike-level data")
        if spike_df is None:
            st.info("No waveform-aligned spike rows are available for this input.")
        else:
            st.dataframe(
                spike_df,
                width="stretch",
                hide_index=True,
                column_config={
                    'SNR (×σ)': st.column_config.NumberColumn(format="%.1f"),
                    'Trough (µV)': st.column_config.NumberColumn(format="%.2f"),
                    'Peak-to-Peak (µV)': st.column_config.NumberColumn(format="%.2f"),
                },
            )
    elif data_view == "Burst data":
        st.subheader("Burst-level data")
        if burst_df.empty:
            st.info("No burst rows are available with the current parameters.")
        else:
            st.dataframe(
                burst_df,
                width="stretch",
                hide_index=True,
                column_config={
                    'Start Time (s)': st.column_config.NumberColumn(format="%.4f"),
                    'End Time (s)': st.column_config.NumberColumn(format="%.4f"),
                    'Duration (ms)': st.column_config.NumberColumn(format="%.1f"),
                    'Mean Amplitude (µV)': st.column_config.NumberColumn(format="%.2f"),
                    'Maximum Amplitude (µV)': st.column_config.NumberColumn(format="%.2f"),
                    'Amplitude SD (µV)': st.column_config.NumberColumn(format="%.2f"),
                    'Amplitude CV': st.column_config.NumberColumn(format="%.3f"),
                    'Attenuation Index': st.column_config.NumberColumn(format="%.3f"),
                    'Mean Spike Width (ms)': st.column_config.NumberColumn(format="%.3f"),
                },
            )
    elif data_view == "Exports & methods":
        st.subheader("Exports")
        export_spikes, export_bursts, export_metadata = st.columns(3)
        if spike_df is not None:
            export_spikes.download_button(
                "Download spike CSV",
                spike_df.to_csv(index=False).encode(),
                "spike_data.csv",
                "text/csv",
            )
        export_bursts.download_button(
            "Download burst CSV",
            burst_df.to_csv(index=False).encode(),
            "burst_data.csv",
            "text/csv",
        )
        export_metadata.download_button(
            "Download metadata JSON",
            analysis_metadata_json.encode(),
            "analysis_metadata.json",
            "application/json",
        )

        st.markdown("### Methods text")
        st.markdown("Generate a draft methods paragraph from the current analysis settings.")

        if st.button("Generate methods text"):
            if burst_method == "Max Interval":
                method_name = "Max Interval"
                citation = "(Cotterill et al. 2016, J Neurophysiol)"
                method_frag = (
                    f"The following parameters were used: maximum beginning ISI = {max_beg_isi} ms, "
                    f"maximum end ISI = {max_end_isi} ms, minimum interburst interval = {min_ibi_ms} ms, "
                    f"minimum burst duration = {min_dur_ms} ms, minimum spikes per burst = {min_spk}."
                )
                compare_frag = ""
            elif burst_method == "Adaptive logISI":
                method_name = "logISI adaptive"
                citation = "(Pasquale et al. 2010, J Comput Neurosci)"
                _ith = f"{isi_th:.1f}" if isi_th is not None else "100.0"
                _vp = f"{void_param:.2f}" if void_param is not None else "N/A"
                _comp = "exceeded" if (isi_th is not None and isi_th > 100) else "did not exceed"
                _mode = "dual-threshold boundary extension" if dual_thresh else "single threshold"
                fb_note = " The method fell back to the fixed 100 ms criterion." if logisi_fb else ""
                method_frag = (
                    f"The ISI threshold was automatically determined from the logarithmic ISI "
                    f"histogram as ISIth = {_ith} ms (void parameter = {_vp}), which {_comp} 100 ms, "
                    f"so {_mode} was applied.{fb_note}"
                )
                compare_frag = ""
            else:
                method_name = "Max Interval (primary) and logISI adaptive (comparison)"
                citation = (
                    "(Cotterill et al. 2016, J Neurophysiol; "
                    "Pasquale et al. 2010, J Comput Neurosci)"
                )
                _ith = f"{isi_th:.1f}" if isi_th is not None else "100.0"
                _vp = f"{void_param:.2f}" if void_param is not None else "N/A"
                agreement_level = (
                    "high" if (hamming_pct is not None and hamming_pct < 5) else
                    "moderate" if (hamming_pct is not None and hamming_pct <= 10) else
                    "low"
                )
                method_frag = (
                    f"For the Max Interval method: maximum beginning ISI = {max_beg_isi} ms, "
                    f"maximum end ISI = {max_end_isi} ms, minimum interburst interval = {min_ibi_ms} ms, "
                    f"minimum burst duration = {min_dur_ms} ms, minimum spikes per burst = {min_spk}. "
                    f"For the logISI method: ISIth = {_ith} ms (void parameter = {_vp})."
                )
                _hd = f"{hamming_pct:.1f}" if hamming_pct is not None else "N/A"
                compare_frag = (
                    f" Both methods were applied and showed {agreement_level} agreement on burst "
                    f"occupancy (normalized Hamming distance = {_hd}%; Cotterill et al. 2016)."
                )

            if has_raw_trace:
                detection_frag = (
                    f"Spikes were detected as negative threshold crossings exceeding {thr_mult}× "
                    f"the estimated noise floor, computed as the median absolute deviation of the "
                    f"bandpass-filtered signal ({int(bp_low)}–{int(bp_high)} Hz, "
                    f"4th-order Butterworth; Quiroga et al. 2004), with a 1 ms refractory period. "
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
            st.warning(METHODS_REVIEW_WARNING)
            st.text_area("Methods section paragraph", methods_text, height=200)

st.markdown("---")
st.caption(
    "SpikeLab — "
    "Cotterill et al. (2016) J Neurophysiol · "
    "Pasquale et al. (2010) J Comput Neurosci"
)
