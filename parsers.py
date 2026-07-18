import re
from io import StringIO

import numpy as np
import pandas as pd


def parse_raw_content(content: bytes):
    """Parse raw voltage trace: two-column [time, voltage], auto-detects tab or comma delimiter."""
    text = content.decode('utf-8', errors='ignore')
    sample = next((ln for ln in text.splitlines() if ln.strip()), '')
    delimiter = ',' if ',' in sample and '\t' not in sample else '\t'

    try:
        df = pd.read_csv(
            StringIO(text), sep=delimiter, header=None, usecols=[0, 1],
            names=['t', 'v'], engine='c', on_bad_lines='skip',
        )
    except (ValueError, pd.errors.EmptyDataError):
        # Empty file, or no row has two columns under the detected delimiter.
        return np.array([]), np.array([])
    # Coerce non-numeric rows (headers, junk) to NaN and drop them — this reproduces
    # the old per-row try/except while parsing the numeric bulk in C, not Python.
    t = pd.to_numeric(df['t'], errors='coerce')
    v = pd.to_numeric(df['v'], errors='coerce')
    keep = t.notna() & v.notna()
    return t[keep].to_numpy(dtype=float), v[keep].to_numpy(dtype=float)


def _decode_edf_header(content: bytes):
    """Parse the fixed + per-signal EDF header (EDF/EDF+, 16-bit signals).

    Returns a dict with labels, physical dimensions, samples-per-record (spr),
    physical/digital ranges, record duration, record count and header size.
    Truncated records and discontinuous EDF+D files are rejected here so no
    caller can accidentally construct a scientifically incorrect regular trace.
    """
    if len(content) < 256:
        raise ValueError("File is smaller than a valid EDF header (256 bytes).")

    head = content[:256]

    def field(a, b):
        return head[a:b].decode('ascii', 'ignore').strip()

    reserved = field(192, 236)
    if reserved.startswith('EDF+D'):
        raise ValueError(
            "Discontinuous EDF+D recordings are not supported because their "
            "record start times cannot be represented as one regular trace."
        )

    try:
        n_records_declared = int(field(236, 244))
        record_duration = float(field(244, 252))
        ns = int(field(252, 256))
    except ValueError as exc:
        raise ValueError("Malformed EDF header (record/signal counts).") from exc
    if ns <= 0:
        raise ValueError("EDF header reports no signals.")
    if record_duration <= 0:
        raise ValueError("EDF header reports a non-positive record duration.")

    header_bytes = 256 + ns * 256
    if len(content) < header_bytes:
        raise ValueError("File is truncated: the per-signal header is incomplete.")

    sig = content[256:header_bytes]

    # Per-signal fields are stored column-wise: all labels, then all transducers,
    # etc. `take` reads one ns-wide column of `size`-byte fields and advances.
    cursor = 0

    def take(size, encoding='ascii'):
        nonlocal cursor
        base = cursor
        cursor += ns * size
        return [sig[base + i * size: base + (i + 1) * size].decode(encoding, 'ignore').strip()
                for i in range(ns)]

    labels = take(16)
    take(80)                    # transducer type (unused)
    physical_dimensions = take(8, encoding='latin-1')
    phys_min = take(8)
    phys_max = take(8)
    dig_min = take(8)
    dig_max = take(8)
    take(80)                    # prefiltering (unused)
    n_samp = take(8)            # samples per data record

    try:
        spr = [int(x) for x in n_samp]
        pmin = [float(x) for x in phys_min]
        pmax = [float(x) for x in phys_max]
        dmin = [float(x) for x in dig_min]
        dmax = [float(x) for x in dig_max]
    except ValueError as exc:
        raise ValueError("Malformed EDF signal header (non-numeric field).") from exc

    if any(x < 0 for x in spr):
        raise ValueError("Malformed EDF signal header (negative samples per record).")
    record_size = sum(spr)                    # int16 values per data record
    if record_size <= 0:
        raise ValueError("EDF header reports no samples in any signal.")
    bytes_per_record = record_size * 2
    data_bytes = len(content) - header_bytes
    available_records, remainder = divmod(data_bytes, bytes_per_record)
    if remainder:
        raise ValueError(
            "File is truncated: the final EDF data record is incomplete "
            f"({remainder} trailing bytes)."
        )
    if n_records_declared > 0 and available_records < n_records_declared:
        raise ValueError(
            "File is truncated: the EDF header declares "
            f"{n_records_declared} data records but only {available_records} are present."
        )
    if n_records_declared > 0 and available_records > n_records_declared:
        raise ValueError(
            "EDF record count mismatch: the header declares "
            f"{n_records_declared} data records but {available_records} are present."
        )
    n_records = available_records if n_records_declared <= 0 else n_records_declared

    return {
        'labels': labels, 'physical_dimensions': physical_dimensions,
        'spr': spr, 'pmin': pmin, 'pmax': pmax,
        'dmin': dmin, 'dmax': dmax, 'record_duration': record_duration,
        'record_size': record_size, 'n_records': n_records, 'header_bytes': header_bytes,
    }


def _is_signal_channel(label, spr):
    """Ordinary data channels only — EDF+ annotation channels carry no samples."""
    return spr > 0 and 'EDF Annotations' not in label


def edf_signal_labels(content: bytes):
    """List EDF signals as (index, label, sampling_rate_hz, n_samples, physical_dimension)."""
    hdr = _decode_edf_header(content)
    out = []
    for i, (lbl, spr) in enumerate(zip(hdr['labels'], hdr['spr'])):
        if not _is_signal_channel(lbl, spr):
            continue
        fs = spr / hdr['record_duration']
        out.append((
            i,
            lbl or f"Signal {i + 1}",
            fs,
            hdr['n_records'] * spr,
            hdr['physical_dimensions'][i],
        ))
    return out


def _edf_dimension_to_uv_factor(dimension):
    """Return the multiplier that converts an EDF physical dimension to µV."""
    normalized = dimension.strip().lower()
    normalized = (
        normalized.replace('î¼', 'u').replace('â', '')
        .replace('µ', 'u').replace('μ', 'u')
    )
    normalized = re.sub(r'[\s._-]+', '', normalized)
    factors = {
        'v': 1_000_000.0,
        'volt': 1_000_000.0,
        'volts': 1_000_000.0,
        'mv': 1_000.0,
        'millivol': 1_000.0,
        'millivolt': 1_000.0,
        'millivolts': 1_000.0,
        'uv': 1.0,
        'microvol': 1.0,
        'microvolt': 1.0,
        'microvolts': 1.0,
    }
    if normalized not in factors:
        declared = dimension or "(blank)"
        raise ValueError(
            f"Unsupported EDF physical dimension {declared!r}; "
            "supported voltage units are V, mV, and µV."
        )
    return factors[normalized]


def parse_edf_content(content: bytes, channel_index: int = None):
    """Read one EDF signal channel into a regular (time_s, voltage_µV) trace.

    Digital int16 samples are rescaled to physical units with the channel's
    own calibration and then normalized from V, mV, or µV to µV. The time
    axis uses that channel's independent sampling rate. Only the requested
    channel is materialised.
    """
    hdr = _decode_edf_header(content)
    signal_channels = [i for i, (lbl, spr) in enumerate(zip(hdr['labels'], hdr['spr']))
                       if _is_signal_channel(lbl, spr)]
    if not signal_channels:
        raise ValueError("The EDF file has no ordinary signal channels.")
    if channel_index is None:
        channel_index = signal_channels[0]
    if channel_index not in signal_channels:
        raise ValueError(f"Channel index {channel_index} is not a selectable signal channel.")
    if hdr['n_records'] <= 0:
        raise ValueError("The EDF file contains no complete data records.")

    spr = hdr['spr']
    record_size = hdr['record_size']
    total = hdr['n_records'] * record_size
    data = np.frombuffer(content, dtype='<i2', count=total, offset=hdr['header_bytes'])
    data = data.reshape(hdr['n_records'], record_size)

    c = channel_index
    start = sum(spr[:c])
    digital = data[:, start:start + spr[c]].reshape(-1).astype(np.float64)

    dmin, dmax = hdr['dmin'][c], hdr['dmax'][c]
    pmin, pmax = hdr['pmin'][c], hdr['pmax'][c]
    if dmax == dmin:
        raise ValueError("EDF channel has a degenerate digital range (digital min == max).")
    physical = (digital - dmin) * (pmax - pmin) / (dmax - dmin) + pmin
    voltage = physical * _edf_dimension_to_uv_factor(hdr['physical_dimensions'][c])

    fs = spr[c] / hdr['record_duration']
    times = np.arange(len(voltage)) / fs
    return times, voltage


def stitch_segments(segments):
    """Sort (t, v) segment arrays by first timestamp and concatenate, trimming overlaps."""
    order = sorted(range(len(segments)), key=lambda i: segments[i][0][0])
    nominal_period = float(np.median(np.concatenate([np.diff(t) for t, _ in segments])))
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
        if not len(kept_t):
            continue
        boundary_interval = float(kept_t[0] - last_t)
        if boundary_interval > 1.5 * nominal_period:
            raise ValueError(
                f"Recording gap detected before segment {idx + 1}: boundary interval "
                f"{boundary_interval:g} s exceeds 1.5 times the expected nominal sample "
                f"period of {nominal_period:g} s."
            )
        stitched_t.append(kept_t)
        stitched_v.append(kept_v)
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


_NE_MULTICHANNEL_FIELD_RE = re.compile(r'"([^"]*)"')
_NE_WAVEFORM_SAMPLES = 76
_NE_CHANNEL_ID_RE = re.compile(
    r'(?:^|[^A-Za-z0-9])Label[_ -]*([A-Za-z]+\d+)(?=[^A-Za-z0-9]|$)',
    re.IGNORECASE,
)


def is_ne_multichannel_content(content: bytes) -> bool:
    """Sniff for the multi-channel NeuroExplorer numeric export: its first line is a
    run of double-quoted, semicolon-delimited column labels — distinct from the
    plain 2-column raw trace and from the single-channel 'Instantaneous Parameters'
    NeuroExplorer export, neither of which quote their header this way."""
    first_line = content.split(b'\n', 1)[0].decode('utf-8', errors='ignore')
    fields = _NE_MULTICHANNEL_FIELD_RE.findall(first_line)
    return len(fields) >= 2 and all(';' in f for f in fields)


def _extract_ne_channel_id(label):
    match = _NE_CHANNEL_ID_RE.search(label)
    return match.group(1).upper() if match else label


def _parse_ne_multichannel_header(header_line: str):
    """Return positional channel groups from a quoted NeuroExplorer header."""
    header_line = header_line.lstrip('\ufeff')
    matches = list(_NE_MULTICHANNEL_FIELD_RE.finditer(header_line))
    if not matches:
        return []
    if header_line[:matches[0].start()] or header_line[matches[-1].end():]:
        raise ValueError("The multichannel header contains text outside quoted fields.")
    for left, right in zip(matches, matches[1:]):
        if header_line[left.end():right.start()] != ' ':
            raise ValueError("Header fields must be separated by one ASCII space.")

    groups = []
    seen_labels = set()
    for column_index, match in enumerate(matches):
        field = match.group(1)
        parts = field.split(';')
        if len(parts) < 2:
            raise ValueError(f"Header column {column_index + 1}: malformed field label.")
        label = parts[0]
        variable = parts[-1].strip()
        if not groups or groups[-1]['label'] != label:
            if label in seen_labels:
                raise ValueError(f"Header column {column_index + 1}: channel {label!r} resumes later.")
            if not re.fullmatch(r'Spi(?:ke?)?_timestamps', variable):
                raise ValueError(
                    f"Header column {column_index + 1}: channel {label!r} must begin "
                    "with Spike_timestamps."
                )
            seen_labels.add(label)
            groups.append({
                'label': label,
                'channel_id': _extract_ne_channel_id(label),
                'start': column_index,
                'waveform_columns': [],
            })
        else:
            value_match = re.fullmatch(r'Spi(?:ke?)?_value_(\d+)', variable)
            if not value_match:
                raise ValueError(
                    f"Header column {column_index + 1}: unexpected variable {variable!r} "
                    f"for channel {label!r}."
                )
            groups[-1]['waveform_columns'].append(int(value_match.group(1)))

    for group_index, group in enumerate(groups):
        group['end'] = groups[group_index + 1]['start'] if group_index + 1 < len(groups) else len(matches)
        waveform_columns = group['waveform_columns']
        if waveform_columns and waveform_columns != list(range(1, _NE_WAVEFORM_SAMPLES + 1)):
            raise ValueError(
                f"Channel {group['label']!r} has a partial waveform header; expected "
                f"Spike_value_1 through Spike_value_{_NE_WAVEFORM_SAMPLES}."
            )
    return groups


def _parse_ne_multichannel_row(line, groups, row_number):
    """Decode NeuroExplorer's positional single-space data layout.

    A waveform group occupies its 77 header-defined positions. A timestamp-only
    group reserves a second empty spacer position in the data row. Keeping that
    otherwise invisible position is what prevents all later groups from shifting.
    """
    tokens = line.split(' ')
    expected_columns = sum(
        group['end'] - group['start'] if group['waveform_columns'] else 2
        for group in groups
    )
    if len(tokens) != expected_columns:
        raise ValueError(
            f"Row {row_number}: found {len(tokens)} positional fields; "
            f"expected {expected_columns} from the header layout."
        )

    fields = []
    position = 0
    for group in groups:
        if group['waveform_columns']:
            width = group['end'] - group['start']
            fields.extend(tokens[position:position + width])
            position += width
        else:
            timestamp_slots = tokens[position:position + 2]
            position += 2
            populated = [value for value in timestamp_slots if value != '']
            if len(populated) > 1:
                raise ValueError(
                    f"Row {row_number}: timestamp-only channel {group['label']!r} "
                    "contains more than one value."
                )
            fields.append(populated[0] if populated else '')
    return fields


def parse_ne_multichannel_content(content: bytes):
    """Parse a NeuroExplorer multi-channel numeric ASCII export: one file bundling
    many channels' worth of pre-sorted spike timestamps (and, for some channels,
    per-spike waveform samples) as separate columns.

    The format uses one literal ASCII space as its field delimiter and reserves
    an additional empty spacer position for timestamp-only channel columns.
    Consecutive spaces are therefore meaningful. Rows are decoded with the
    header-derived group widths and validated channel by channel.

    Only channels with at least one spike are returned — empty channel/unit slots
    are common in these exports (a channel with no sorted units still gets a
    column) and aren't useful downstream. Returns a list of dicts, each
    `{'channel_id': str, 'label': str, 'spike_times': ndarray,
    'waveforms': ndarray | None}`, in the file's original channel order.
    """
    lines = content.decode('utf-8', errors='ignore').splitlines()
    if not lines:
        raise ValueError("The file is empty.")

    groups = _parse_ne_multichannel_header(lines[0])
    if not groups:
        raise ValueError("Could not find any quoted channel columns in the header.")
    data_rows = [(row_number, line) for row_number, line in enumerate(lines[1:], start=2) if line != '']
    if not data_rows:
        raise ValueError("The file has a header but no data rows.")
    timestamps = [[] for _ in groups]
    waveforms = [([] if group['waveform_columns'] else None) for group in groups]
    ended = [False] * len(groups)

    for row_number, line in data_rows:
        fields = _parse_ne_multichannel_row(line, groups, row_number)

        for group_index, group in enumerate(groups):
            block = fields[group['start']:group['end']]
            present = [value != '' for value in block]
            if not any(present):
                ended[group_index] = True
                continue
            if ended[group_index]:
                raise ValueError(
                    f"Row {row_number}: channel {group['label']!r} resumed after an empty row."
                )
            if not all(present):
                raise ValueError(
                    f"Row {row_number}: channel {group['label']!r} has a partial "
                    "timestamp/waveform block."
                )
            try:
                values = np.asarray([float(value) for value in block], dtype=float)
            except ValueError as exc:
                raise ValueError(
                    f"Row {row_number}: channel {group['label']!r} contains a non-numeric value."
                ) from exc
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Row {row_number}: channel {group['label']!r} contains a non-finite value."
                )
            timestamp = values[0]
            if timestamps[group_index] and timestamp <= timestamps[group_index][-1]:
                raise ValueError(
                    f"Row {row_number}: channel {group['label']!r} timestamps are not "
                    "strictly increasing."
                )
            timestamps[group_index].append(timestamp)
            if waveforms[group_index] is not None:
                waveforms[group_index].append(values[1:])

    channels = []
    for group_index, group in enumerate(groups):
        if not timestamps[group_index]:
            continue
        wf = np.asarray(waveforms[group_index], dtype=float) if waveforms[group_index] is not None else None
        channels.append({
            'channel_id': group['channel_id'],
            'label': group['label'],
            'spike_times': np.asarray(timestamps[group_index], dtype=float),
            'waveforms': wf,
        })
    return channels
