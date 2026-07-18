import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(signal, fs, low=300, high=3000, order=4):
    if len(signal) < 2:
        raise ValueError("Trace must contain at least two samples before filtering.")
    if fs <= 0:
        raise ValueError("Sampling rate must be positive.")
    if low <= 0 or high <= 0:
        raise ValueError("Bandpass cutoffs must be positive.")
    if low >= high:
        raise ValueError("Bandpass low cutoff must be lower than high cutoff.")
    nyquist = fs / 2
    if high >= nyquist:
        raise ValueError(
            f"Bandpass high cutoff ({high:g} Hz) must be below the Nyquist frequency ({nyquist:g} Hz)."
        )
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        raise ValueError(
            f"Trace is too short for zero-phase filtering: need more than {padlen} samples."
        )
    return filtfilt(b, a, signal)


def detect_spikes(raw_t, raw_filt, threshold_multiplier, fs):
    """Detect negative threshold crossings with 1ms refractory blanking."""
    noise_floor = np.median(np.abs(raw_filt)) / 0.6745
    threshold   = -threshold_multiplier * noise_floor
    ref_samples = int(0.001 * fs)

    # Vectorised downward crossings: sample below threshold, previous at/above.
    crossings = np.flatnonzero((raw_filt[1:] < threshold) & (raw_filt[:-1] >= threshold)) + 1

    # Enforce the refractory period by keeping only crossings that fall more than
    # ref_samples after the last accepted spike (loop is over crossings, not samples).
    spike_idxs = []
    last = -ref_samples - 1
    for i in crossings:
        if (i - last) > ref_samples:
            spike_idxs.append(int(i))
            last = i

    spike_idxs = np.array(spike_idxs, dtype=int)
    spike_times = raw_t[spike_idxs] if len(spike_idxs) else np.array([])
    return spike_times, spike_idxs, threshold, noise_floor


def detect_bursts(spike_times, max_isi_ms, min_spikes):
    """Simple single-threshold Max Interval burst detection (kept for backwards compatibility)."""
    if len(spike_times) < 2:
        return []
    isis = np.diff(spike_times) * 1000
    bursts = []
    i = 0
    while i < len(spike_times) - 1:
        if isis[i] <= max_isi_ms:
            start = i
            j = i + 1
            while j < len(spike_times) - 1 and isis[j] <= max_isi_ms:
                j += 1
            n = j - start + 1
            if n >= min_spikes:
                bursts.append({
                    'start':    spike_times[start],
                    'end':      spike_times[j],
                    'n_spikes': n,
                    'duration': (spike_times[j] - spike_times[start]) * 1000,
                    'idxs':     list(range(start, j + 1)),
                })
            i = j + 1
        else:
            i += 1
    return bursts


# ── Advanced burst detection (Cotterill et al. 2016; Pasquale et al. 2010) ───

def _gaussian_smooth(arr, sigma=1.0, truncate=3.0):
    """Gaussian smoothing via numpy convolve (no scipy.ndimage required)."""
    radius = max(1, int(truncate * sigma + 0.5))
    x      = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr.astype(float), kernel, mode='same')


def _burst_single_thresh(spike_times, max_isi_s, min_spikes):
    """Single-threshold burst detection (internal helper, times in seconds)."""
    if len(spike_times) < 2:
        return []
    isis = np.diff(spike_times)
    bursts, i = [], 0
    while i < len(spike_times) - 1:
        if isis[i] <= max_isi_s:
            start = i
            j = i + 1
            while j < len(spike_times) - 1 and isis[j] <= max_isi_s:
                j += 1
            n = j - start + 1
            if n >= min_spikes:
                bursts.append({
                    'start':    spike_times[start],
                    'end':      spike_times[j],
                    'n_spikes': n,
                    'duration': (spike_times[j] - spike_times[start]) * 1000,
                    'idxs':     list(range(start, j + 1)),
                })
            i = j + 1
        else:
            i += 1
    return bursts


def _burst_dual_thresh(spike_times, core_isi_s, ext_isi_s, min_spikes):
    """Dual-threshold burst detection for ISIth > 100 ms (Pasquale et al. 2010).
    Finds cores using core_isi_s (100 ms), then extends boundaries with ext_isi_s."""
    cores = _burst_single_thresh(spike_times, core_isi_s, 2)
    if not cores:
        return []
    extended = []
    for core in cores:
        s = np.searchsorted(spike_times, core['start'])
        e = np.searchsorted(spike_times, core['end'])
        while s > 0 and (spike_times[s] - spike_times[s - 1]) <= ext_isi_s:
            s -= 1
        while e < len(spike_times) - 1 and (spike_times[e + 1] - spike_times[e]) <= ext_isi_s:
            e += 1
        n = e - s + 1
        if n >= min_spikes:
            extended.append({
                'start':    spike_times[s],
                'end':      spike_times[e],
                'n_spikes': n,
                'duration': (spike_times[e] - spike_times[s]) * 1000,
                'idxs':     list(range(s, e + 1)),
            })
    # De-overlap
    final = []
    for b in extended:
        if not final or b['start'] > final[-1]['end']:
            final.append(b)
        elif b['end'] > final[-1]['end']:
            s = np.searchsorted(spike_times, final[-1]['start'])
            e = np.searchsorted(spike_times, b['end'])
            final[-1] = {
                'start':    spike_times[s],
                'end':      spike_times[e],
                'n_spikes': e - s + 1,
                'duration': (spike_times[e] - spike_times[s]) * 1000,
                'idxs':     list(range(s, e + 1)),
            }
    return final


def max_interval_method(spike_times, max_beg_isi=0.170, max_end_isi=0.300,
                         min_ibi=0.200, min_duration=0.010, min_spikes=3):
    """Full 5-parameter Max Interval burst detection (Cotterill et al. 2016, Table 1).
    All time parameters in seconds."""
    if len(spike_times) < min_spikes:
        return []
    isis = np.diff(spike_times)
    raw, i = [], 0
    while i < len(isis):
        if isis[i] <= max_beg_isi:
            s = i
            j = i + 1
            while j < len(isis) and isis[j] <= max_end_isi:
                j += 1
            raw.append([s, j])
            i = j + 1
        else:
            i += 1
    if not raw:
        return []
    merged = [raw[0][:]]
    for burst in raw[1:]:
        ibi = spike_times[burst[0]] - spike_times[merged[-1][1]]
        if ibi < min_ibi:
            merged[-1][1] = burst[1]
        else:
            merged.append(burst[:])
    bursts = []
    for s_idx, e_idx in merged:
        n     = e_idx - s_idx + 1
        dur_s = spike_times[e_idx] - spike_times[s_idx]
        if n >= min_spikes and dur_s >= min_duration:
            bursts.append({
                'start':    spike_times[s_idx],
                'end':      spike_times[e_idx],
                'n_spikes': n,
                'duration': dur_s * 1000,
                'idxs':     list(range(s_idx, e_idx + 1)),
            })
    return bursts


def logisi_method(spike_times, min_spikes=3, void_thresh=0.7):
    """Pasquale et al. 2010 logISI adaptive burst detection.

    Returns (bursts, isi_th_ms, void_param, fallback, hist_data).
    hist_data keys: bin_centers (log10 ms), counts, smoothed, peaks, p1, p2,
                    isi_th_ms, void_param.
    """
    _empty = {
        'bin_centers': np.array([]), 'counts': np.array([]),
        'smoothed': np.array([]),    'peaks': np.array([]),
        'p1': None, 'p2': None,     'isi_th_ms': 100.0, 'void_param': 0.0,
    }
    if len(spike_times) < min_spikes + 1:
        return [], 100.0, 0.0, True, _empty

    isis_ms = np.diff(spike_times) * 1000.0
    isis_ms = isis_ms[isis_ms > 0]
    if len(isis_ms) < 10:
        return [], 100.0, 0.0, True, _empty

    log_isis  = np.log10(isis_ms)
    bin_size  = 0.1
    bin_min   = np.floor(log_isis.min() * 10) / 10
    bin_max   = np.ceil(log_isis.max()  * 10) / 10 + bin_size
    bin_edges = np.arange(bin_min, bin_max, bin_size)
    if len(bin_edges) < 3:
        bin_edges = np.linspace(log_isis.min(), log_isis.max() + 0.1, 20)

    counts, bin_edges = np.histogram(log_isis, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    smoothed    = _gaussian_smooth(counts, sigma=1.0)

    peaks, _ = (find_peaks(smoothed, height=0.5) if len(smoothed) >= 3
                else (np.array([]), {}))

    p1_idx = p2_idx = None
    isi_th_ms = void_param = None
    fallback  = False

    if len(peaks) >= 2:
        intra = peaks[bin_centers[peaks] < 2.0]   # < 100 ms
        inter = peaks[bin_centers[peaks] >= 2.0]  # >= 100 ms
        if len(intra) > 0 and len(inter) > 0:
            p1_idx = int(intra[np.argmax(smoothed[intra])])
            p2_idx = int(inter[np.argmax(smoothed[inter])])
            if p1_idx > p2_idx:
                p1_idx, p2_idx = p2_idx, p1_idx
            region  = smoothed[p1_idx: p2_idx + 1]
            min_idx = p1_idx + int(np.argmin(region))
            g_min, g1, g2 = smoothed[min_idx], smoothed[p1_idx], smoothed[p2_idx]
            denom = np.sqrt(g1 * g2)
            if denom > 0:
                void_param = float(1.0 - g_min / denom)
                if void_param >= void_thresh:
                    isi_th_ms = float(10 ** bin_centers[min_idx])

    if isi_th_ms is None:
        isi_th_ms  = 100.0
        void_param = void_param if void_param is not None else 0.0
        fallback   = True

    hist_data = {
        'bin_centers': bin_centers,
        'counts':      counts,
        'smoothed':    smoothed,
        'peaks':       peaks,
        'p1':          p1_idx,
        'p2':          p2_idx,
        'isi_th_ms':   isi_th_ms,
        'void_param':  void_param,
    }

    if not fallback and isi_th_ms > 100.0:
        bursts = _burst_dual_thresh(spike_times, 0.100, isi_th_ms / 1000.0, min_spikes)
    else:
        bursts = _burst_single_thresh(spike_times, isi_th_ms / 1000.0, min_spikes)

    return bursts, isi_th_ms, void_param, fallback, hist_data


def compare_methods(bursts_mi, bursts_logisi, recording_duration, bin_size=0.050):
    """Normalised Hamming distance between MI and logISI burst calls (Cotterill et al. 2016)."""
    n_bins = max(1, int(np.ceil(recording_duration / bin_size)))

    def to_binary(bursts):
        v = np.zeros(n_bins, dtype=np.int8)
        for b in bursts:
            s = int(b['start'] / bin_size)
            e = min(int(b['end'] / bin_size) + 1, n_bins)
            if s < n_bins:
                v[s:e] = 1
        return v

    v_mi  = to_binary(bursts_mi)
    v_log = to_binary(bursts_logisi)
    hamming_pct = float(np.sum(v_mi != v_log)) / n_bins * 100.0

    if hamming_pct < 5:
        label = "Strong agreement ✅"
    elif hamming_pct <= 10:
        label = "Moderate agreement ⚠️"
    else:
        label = "Poor agreement ❌ — review parameters"

    return hamming_pct, label


def waveform_amplitude_stats(waveforms):
    """Per-waveform trough, peak, and peak-to-peak amplitude from a (n_spikes, n_samples) matrix."""
    if len(waveforms) == 0:
        return np.array([]), np.array([]), np.array([])
    troughs = waveforms.min(axis=1)
    peaks = waveforms.max(axis=1)
    return troughs, peaks, peaks - troughs


def extract_waveforms(raw_t, raw_filt, spike_times, fs, pre_ms=1.0, post_ms=2.0):
    pre  = int(pre_ms  / 1000 * fs)
    post = int(post_ms / 1000 * fs)
    waveforms, valid_times = [], []
    for st in spike_times:
        idx = np.searchsorted(raw_t, st)
        if idx - pre >= 0 and idx + post < len(raw_filt):
            waveforms.append(raw_filt[idx - pre: idx + post])
            valid_times.append(st)
    waveforms = np.array(waveforms)
    troughs, peaks, p2p = waveform_amplitude_stats(waveforms)
    t_axis = np.arange(-pre, post) / fs * 1000
    return waveforms, troughs, peaks, p2p, np.array(valid_times), t_axis


def build_summary_df(spike_times, bursts, troughs, p2p, valid_times, noise_floor, in_burst_mask=None):
    rows = []
    if in_burst_mask is None:
        burst_spike_set = set()
        for b in bursts:
            for idx in b['idxs']:
                burst_spike_set.add(idx)
    else:
        in_burst_mask = np.asarray(in_burst_mask, dtype=bool)
        if len(in_burst_mask) != len(valid_times):
            raise ValueError("in_burst_mask must have the same length as valid_times.")
    for i, (st, tr, pp) in enumerate(zip(valid_times, troughs, p2p)):
        in_burst = bool(in_burst_mask[i]) if in_burst_mask is not None else i in burst_spike_set
        rows.append({
            'Spike #':             i + 1,
            'Time (s)':            round(float(st), 5),
            'Trough (µV)':         round(float(tr), 2),
            'Peak-to-Peak (µV)':   round(float(pp), 2),
            'SNR (×σ)':            round(abs(float(tr)) / noise_floor, 1),
            'In Burst':            'Yes' if in_burst else 'No',
        })
    return pd.DataFrame(rows)


def compute_spike_widths(waveforms, t_axis):
    """Trough-to-peak width (ms): trough = global min sample, peak = max sample
    after the trough. NaN if the trough is the last sample."""
    n = len(waveforms)
    widths = np.full(n, np.nan)
    for i, w in enumerate(waveforms):
        trough_idx = int(np.argmin(w))
        if trough_idx < len(w) - 1:
            peak_idx = trough_idx + int(np.argmax(w[trough_idx:]))
            widths[i] = t_axis[peak_idx] - t_axis[trough_idx]
    return widths


def compute_isi_arrays(valid_times):
    """Preceding/following ISI (ms), aligned to valid_times. Edge spikes get NaN."""
    n = len(valid_times)
    isi_pre, isi_post = np.full(n, np.nan), np.full(n, np.nan)
    if n > 1:
        d = np.diff(valid_times) * 1000.0
        isi_pre[1:], isi_post[:-1] = d, d
    return isi_pre, isi_post


def _burst_valid_slice(b, valid_times):
    """Time-based match of a burst's [start, end] to a valid_times index slice."""
    s = np.searchsorted(valid_times, b['start'], side='left')
    e = np.searchsorted(valid_times, b['end'],   side='right')
    return int(s), int(e)


def compute_burst_amplitude_stats(bursts, valid_times, p2p, widths=None):
    """Per-burst amplitude dynamics. Returns list of dicts = shallow copies of
    each burst dict plus valid_idx_start/end, n_valid_spikes, mean_amp, max_amp,
    sd_amp, cv_amp, attenuation_index, mean_width.
    attenuation_index = (first_valid_amp - last_valid_amp) / first_valid_amp.
    Needs >=2 valid spikes for sd/cv/attenuation, else NaN."""
    stats = []
    for b in bursts:
        s, e = _burst_valid_slice(b, valid_times)
        amps = p2p[s:e]
        entry = dict(b)
        entry.update(valid_idx_start=s, valid_idx_end=e, n_valid_spikes=len(amps))
        entry['mean_amp'] = float(np.mean(amps)) if len(amps) >= 1 else np.nan
        entry['max_amp']  = float(np.max(amps))  if len(amps) >= 1 else np.nan
        if len(amps) >= 2:
            entry['sd_amp'] = float(np.std(amps, ddof=1))
            entry['cv_amp'] = entry['sd_amp'] / entry['mean_amp'] if entry['mean_amp'] else np.nan
            first, last = amps[0], amps[-1]
            entry['attenuation_index'] = float((first - last) / first) if first != 0 else np.nan
        else:
            entry['sd_amp'] = entry['cv_amp'] = entry['attenuation_index'] = np.nan
        entry['mean_width'] = (float(np.nanmean(widths[s:e]))
                                if widths is not None and len(amps) >= 1 else np.nan)
        stats.append(entry)
    return stats


def compute_intraburst_decrement(burst_stats, p2p):
    """Spike-position-within-burst vs amplitude, flattened across bursts with
    >=2 matched valid spikes, for the aggregated decrement plot."""
    positions, amps, burst_ids = [], [], []
    for bid, b in enumerate(burst_stats):
        s, e = b['valid_idx_start'], b['valid_idx_end']
        n = e - s
        if n >= 2:
            positions.extend(range(n))
            amps.extend(p2p[s:e])
            burst_ids.extend([bid] * n)
    return np.array(positions), np.array(amps), np.array(burst_ids)


def burst_membership_mask(burst_stats, n_valid):
    """Boolean mask over valid_times: True if the spike falls inside any burst's
    time-matched window. Time-based — independent of build_summary_df's existing
    idx-based 'In Burst' column, which stays untouched."""
    mask = np.zeros(n_valid, dtype=bool)
    for b in burst_stats:
        mask[b['valid_idx_start']:b['valid_idx_end']] = True
    return mask
