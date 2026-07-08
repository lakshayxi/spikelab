import numpy as np


def parse_raw_content(content: bytes):
    """Parse raw voltage trace: two-column [time, voltage], auto-detects tab or comma delimiter."""
    text = content.decode('utf-8', errors='ignore')
    sample = next((l for l in text.splitlines() if l.strip()), '')
    delimiter = ',' if ',' in sample and '\t' not in sample else '\t'
    times, voltages = [], []
    for line in text.splitlines():
        parts = line.strip().split(delimiter)
        if len(parts) == 2:
            try:
                times.append(float(parts[0]))
                voltages.append(float(parts[1]))
            except ValueError:
                pass
    return np.array(times), np.array(voltages)


def stitch_segments(segments):
    """Sort (t, v) segment arrays by first timestamp and concatenate, trimming overlaps."""
    order = sorted(range(len(segments)), key=lambda i: segments[i][0][0])
    t0, v0 = segments[order[0]]
    stitched_t, stitched_v = [t0], [v0]
    summary = [{'segment_index': order[0], 'first_t': float(t0[0]), 'last_t': float(t0[-1]),
                'n_samples': len(t0), 'n_kept': len(t0), 'overlap_s': 0.0, 'overlap_samples': 0}]
    last_t = t0[-1]
    for idx in order[1:]:
        t, v = segments[idx]
        cut = np.searchsorted(t, last_t, side='right')
        kept_t, kept_v = t[cut:], v[cut:]
        summary.append({'segment_index': idx, 'first_t': float(t[0]), 'last_t': float(t[-1]),
                         'n_samples': len(t), 'n_kept': len(kept_t),
                         'overlap_s': float(max(0.0, last_t - t[0])), 'overlap_samples': int(cut)})
        stitched_t.append(kept_t)
        stitched_v.append(kept_v)
        if len(kept_t):
            last_t = kept_t[-1]
    return np.concatenate(stitched_t), np.concatenate(stitched_v), summary


def parse_neuroexplorer_content(content: bytes):
    """Parse NeuroExplorer ASCII export — returns spike timestamps, ISIs, and firing rates."""
    spike_times, isis, freqs = [], [], []
    in_instant = False
    for line in content.decode('utf-8', errors='ignore').splitlines():
        line = line.strip()
        if 'Instantaneous Parameters' in line:
            in_instant = True
            continue
        if 'Binned Parameters' in line:
            in_instant = False
            continue
        if in_instant:
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    t   = float(parts[0])
                    isi = float(parts[1]) if parts[1] not in ('NaN', '') else np.nan
                    fr  = float(parts[2]) if parts[2] not in ('NaN', '') else np.nan
                    spike_times.append(t)
                    isis.append(isi)
                    freqs.append(fr)
                except ValueError:
                    pass
    return np.array(spike_times), np.array(isis), np.array(freqs)
