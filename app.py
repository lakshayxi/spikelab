import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use('Agg')
import warnings
from io import BytesIO

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

warnings.filterwarnings('ignore')

from parsers import parse_neuroexplorer_content, parse_raw_content, stitch_segments  # noqa: E402
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
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MEA Spike Analyser",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Plot colour palette (light-theme scientific) ───────────────────────────────
C_BG    = '#FFFFFF'
C_PANEL = '#F8F9FA'
C_GRID  = '#E5E7EB'
C_TEXT  = '#111827'
C_MUTED = '#6B7280'
C_SPIKE = '#2563EB'
C_BURST = '#DC2626'
C_LOGISI = '#7C3AED'
C_ANNOT = '#92400E'   # dark amber for annotations

def style_ax(ax, title=''):
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(C_GRID)
    ax.xaxis.label.set_color(C_TEXT)
    ax.yaxis.label.set_color(C_TEXT)
    if title:
        ax.set_title(title, color=C_TEXT, fontsize=10, fontweight='bold', pad=6)
    ax.grid(color=C_GRID, linewidth=0.4, alpha=0.8)

def _strip_emoji(s):
    """Remove trailing status emojis from compare_methods() return strings."""
    return s.replace('✅', '').replace('⚠️', '').replace('❌', '').strip()

# ── Cached parsing wrappers ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def parse_raw_file(content: bytes):
    return parse_raw_content(content)

@st.cache_data(show_spinner=False)
def parse_neuroexplorer_file(content: bytes):
    return parse_neuroexplorer_content(content)

# ── Cached burst wrappers (spike_times as tuple for cache hashing) ────────────
@st.cache_data(show_spinner=False)
def _cached_mi(spike_t, beg, end, ibi, dur, spk):
    return max_interval_method(np.array(spike_t), beg, end, ibi, dur, spk)

@st.cache_data(show_spinner=False)
def _cached_logisi(spike_t, min_spk, void_th):
    return logisi_method(np.array(spike_t), min_spk, void_th)

# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════
def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=C_BG)
    buf.seek(0)
    return buf

def panel_to_bytes(fig, axes):
    """Crop and export just the region of one or more Axes (e.g. a panel plus its
    colorbar) from an already-drawn multi-panel figure — guarantees the individual
    export is pixel-identical to that panel in the combined download."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = Bbox.union([ax.get_tightbbox(renderer) for ax in axes]) \
               .transformed(fig.dpi_scale_trans.inverted())
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches=bbox, facecolor=C_BG)
    buf.seek(0)
    return buf

def render_panel_downloads(fig, panels, base_filename):
    """Compact row of small download buttons, one per panel of a multi-panel figure."""
    if not panels:
        return
    cols = st.columns(len(panels))
    for col, (short_name, label, axes) in zip(cols, panels):
        col.download_button(
            "⬇️", panel_to_bytes(fig, axes), f"{base_filename}_{short_name}.png",
            "image/png", help=f"Download: {label}", key=f"dl_{base_filename}_{short_name}",
        )

def plot_overview(spike_times, bursts, freqs):
    fig = plt.figure(figsize=(14, 6), facecolor=C_BG)
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45,
                            left=0.06, right=0.98, top=0.90, bottom=0.10)
    fig.suptitle('Spike Raster and Firing Rate', color=C_TEXT, fontsize=13, fontweight='bold')

    ax1 = fig.add_subplot(gs[0])
    style_ax(ax1, 'Spike Raster')
    for b in bursts:
        ax1.axvspan(b['start'], b['end'], color=C_BURST, alpha=0.15)
    ax1.eventplot(spike_times, lineoffsets=0, linelengths=0.7,
                  color=C_SPIKE, linewidths=1.2)
    for k, b in enumerate(bursts):
        mid = (b['start'] + b['end']) / 2
        ax1.text(mid, 0.55, f"B{k+1}", color=C_BURST, fontsize=7, ha='center',
                 fontweight='bold', transform=ax1.get_xaxis_transform())
    ax1.set_yticks([])
    ax1.set_xlim(spike_times[0] - 0.3, spike_times[-1] + 0.3)
    ax1.set_ylabel('Spikes', fontsize=9)

    ax2 = fig.add_subplot(gs[1])
    style_ax(ax2, 'Instantaneous Firing Rate')
    valid = ~np.isnan(freqs[:len(spike_times)])
    if valid.any():
        ax2.fill_between(spike_times[valid], freqs[:len(spike_times)][valid],
                         alpha=0.2, color=C_SPIKE)
        ax2.plot(spike_times[valid], freqs[:len(spike_times)][valid],
                 color=C_SPIKE, linewidth=0.9)
    for b in bursts:
        ax2.axvspan(b['start'], b['end'], color=C_BURST, alpha=0.10)
    ax2.set_xlabel('Time (s)', fontsize=9)
    ax2.set_ylabel('Freq (Hz)', fontsize=9)
    ax2.set_xlim(spike_times[0] - 0.3, spike_times[-1] + 0.3)
    panels = [
        ('raster',       'Spike Raster',               [ax1]),
        ('firing_rate',  'Instantaneous Firing Rate',  [ax2]),
    ]
    return fig, panels

def plot_isi(spike_times, isi_threshold_ms, secondary_threshold_ms=None):
    isis = np.diff(spike_times) * 1000
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=C_BG)
    fig.suptitle('Inter-Spike Interval Analysis', color=C_TEXT, fontsize=13, fontweight='bold')

    bins    = np.linspace(0, min(isis.max(), 1000), 60)
    centers = (bins[:-1] + bins[1:]) / 2
    counts, _ = np.histogram(isis, bins=bins)
    colors  = [C_SPIKE if c <= isi_threshold_ms else C_MUTED for c in centers]

    for ax, ylabel, yscale in [(axes[0], 'Count', 'linear'), (axes[1], 'Count (log)', 'log')]:
        title = 'ISI Histogram' if yscale == 'linear' else 'ISI Histogram (log scale)'
        style_ax(ax, title)
        y = counts if yscale == 'linear' else counts + 0.5
        ax.bar(centers, y, width=bins[1] - bins[0], color=colors, edgecolor='none', alpha=0.80)
        ax.axvline(isi_threshold_ms, color=C_BURST, linewidth=1.5, linestyle='--',
                   label=f'ISIth = {isi_threshold_ms:.0f} ms')
        if secondary_threshold_ms is not None:
            ax.axvline(secondary_threshold_ms, color=C_LOGISI, linewidth=1.5, linestyle=':',
                       label=f'Extension threshold = {secondary_threshold_ms:.0f} ms')
        if yscale == 'log':
            ax.set_yscale('log')
        ax.set_xlabel('ISI (ms)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8, facecolor=C_PANEL, edgecolor=C_GRID)

    fig.tight_layout()
    panels = [
        ('linear', 'ISI Histogram',            [axes[0]]),
        ('log',    'ISI Histogram (log scale)', [axes[1]]),
    ]
    return fig, panels

def plot_amplitude(waveforms, troughs, peaks, p2p, valid_times, t_axis, noise_floor, threshold):
    fig = plt.figure(figsize=(14, 10), facecolor=C_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32,
                            left=0.07, right=0.97, top=0.92, bottom=0.07)
    fig.suptitle('Spike Amplitude Quantification', color=C_TEXT, fontsize=13, fontweight='bold')
    cmap     = plt.cm.viridis
    norm_amp = ((np.abs(troughs) - np.abs(troughs).min()) /
                (np.abs(troughs).max() - np.abs(troughs).min() + 1e-9))

    ax_A = fig.add_subplot(gs[0, 0])
    style_ax(ax_A, 'A  All Spike Waveforms')
    for w, na in zip(waveforms, norm_amp):
        ax_A.plot(t_axis, w, color=cmap(na), alpha=0.20, linewidth=0.5)
    mean_w = waveforms.mean(axis=0)
    ax_A.plot(t_axis, mean_w, color=C_TEXT, linewidth=2.0, label='Mean waveform', zorder=5)
    ax_A.axhline(threshold, color=C_BURST, linewidth=1.0, linestyle='--',
                 label=f'Threshold ({threshold:.1f} µV)', alpha=0.8)
    ax_A.axhline(0, color=C_MUTED, linewidth=0.4)
    ax_A.axvline(0, color=C_MUTED, linewidth=0.4, linestyle=':')
    ti = t_axis[mean_w.argmax()]
    ax_A.annotate('', xy=(ti, mean_w.max()), xytext=(ti, mean_w.min()),
                  arrowprops=dict(arrowstyle='<->', color=C_ANNOT, lw=1.5))
    ax_A.text(ti + 0.08, (mean_w.max() + mean_w.min()) / 2,
              f'{mean_w.max() - mean_w.min():.1f} µV', color=C_ANNOT, fontsize=8, va='center')
    ax_A.set_xlabel('Time (ms)', fontsize=9)
    ax_A.set_ylabel('Voltage (µV)', fontsize=9)
    ax_A.legend(fontsize=8, facecolor=C_PANEL, edgecolor=C_GRID)
    sm = plt.cm.ScalarMappable(cmap=cmap,
         norm=plt.Normalize(np.abs(troughs).min(), np.abs(troughs).max()))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_A, pad=0.02)
    cb.set_label('|Trough| (µV)', color=C_TEXT, fontsize=8)
    cb.ax.tick_params(colors=C_TEXT, labelsize=8)

    ax_B = fig.add_subplot(gs[0, 1])
    style_ax(ax_B, 'B  Trough Amplitude Distribution')
    n_bins = min(25, max(8, len(troughs) // 3))
    _, bins, patches = ax_B.hist(troughs, bins=n_bins, edgecolor='none', alpha=0.80)
    for patch, left in zip(patches, bins[:-1]):
        na = (abs(left) - abs(troughs).min()) / (abs(troughs).max() - abs(troughs).min() + 1e-9)
        patch.set_facecolor(cmap(na))
    ax_B.axvline(troughs.mean(),     color=C_TEXT,  lw=1.5, ls='--',
                 label=f'Mean: {troughs.mean():.1f} µV')
    ax_B.axvline(np.median(troughs), color=C_ANNOT, lw=1.5, ls=':',
                 label=f'Median: {np.median(troughs):.1f} µV')
    ax_B.axvline(threshold, color=C_BURST, lw=1.2, ls='-.',
                 label=f'Threshold: {threshold:.1f} µV')
    ax_B.set_xlabel('Trough amplitude (µV)', fontsize=9)
    ax_B.set_ylabel('Count', fontsize=9)
    ax_B.legend(fontsize=8, facecolor=C_PANEL, edgecolor=C_GRID)

    ax_C = fig.add_subplot(gs[1, 0])
    style_ax(ax_C, 'C  Peak-to-Peak Amplitude Distribution')
    ax_C.hist(p2p, bins=n_bins, color=C_BURST, alpha=0.75, edgecolor='none')
    ax_C.axvline(p2p.mean(),     color=C_TEXT,  lw=1.5, ls='--',
                 label=f'Mean: {p2p.mean():.1f} µV')
    ax_C.axvline(np.median(p2p), color=C_ANNOT, lw=1.5, ls=':',
                 label=f'Median: {np.median(p2p):.1f} µV')
    ax_C.set_xlabel('Peak-to-peak (µV)', fontsize=9)
    ax_C.set_ylabel('Count', fontsize=9)
    ax_C.legend(fontsize=8, facecolor=C_PANEL, edgecolor=C_GRID)

    ax_D = fig.add_subplot(gs[1, 1])
    style_ax(ax_D, 'D  Spike Amplitude Over Time')
    sc = ax_D.scatter(valid_times, p2p, c=p2p, cmap=cmap, s=35, alpha=0.80, zorder=3)
    ax_D.axhline(p2p.mean(), color=C_TEXT, lw=1.0, ls='--', alpha=0.7,
                 label=f'Mean P2P: {p2p.mean():.1f} µV')
    if len(valid_times) >= 2:
        slope, intercept = np.polyfit(valid_times, p2p, 1)
        ax_D.plot(valid_times, slope * valid_times + intercept, color=C_ANNOT, lw=1.6,
                  label=f'Trend: {slope:+.2f} µV/s', zorder=4)
    ax_D.set_xlabel('Time (s)', fontsize=9)
    ax_D.set_ylabel('Peak-to-peak (µV)', fontsize=9)
    ax_D.legend(fontsize=8, facecolor=C_PANEL, edgecolor=C_GRID)
    cb2 = fig.colorbar(sc, ax=ax_D, pad=0.02)
    cb2.set_label('P2P (µV)', color=C_TEXT, fontsize=8)
    cb2.ax.tick_params(colors=C_TEXT, labelsize=8)
    panels = [
        ('A_waveforms',    'A — All Spike Waveforms',                 [ax_A, cb.ax]),
        ('B_trough_dist',  'B — Trough Amplitude Distribution',       [ax_B]),
        ('C_p2p_dist',     'C — Peak-to-Peak Amplitude Distribution', [ax_C]),
        ('D_amp_time',     'D — Spike Amplitude Over Time',           [ax_D, cb2.ax]),
    ]
    return fig, panels

def plot_bursts(spike_times, bursts, p2p, valid_times):
    if not bursts:
        return None, []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=C_BG)
    fig.suptitle('Burst Analysis', color=C_TEXT, fontsize=13, fontweight='bold')

    durs = [b['duration'] for b in bursts]
    ns   = [b['n_spikes'] for b in bursts]
    x    = np.arange(len(bursts))
    lbls = [f"B{k+1}" for k in range(len(bursts))]

    ax = axes[0]
    style_ax(ax, 'Per-Burst: Duration and Spike Count')
    ax2 = ax.twinx()
    ax.bar(x - 0.2, durs, 0.35, color=C_BURST, alpha=0.75, label='Duration (ms)')
    ax2.bar(x + 0.2, ns,   0.35, color=C_SPIKE, alpha=0.75, label='Spike count')
    ax.set_xticks(x)
    ax.set_xticklabels(lbls)
    ax.set_ylabel('Duration (ms)', fontsize=9)
    ax2.set_ylabel('Spike count', fontsize=9)
    ax.tick_params(colors=C_TEXT)
    ax2.tick_params(colors=C_TEXT)
    ax2.yaxis.label.set_color(C_TEXT)
    for sp in ax2.spines.values():
        sp.set_color(C_GRID)
    l1, h1 = ax.get_legend_handles_labels()
    l2, h2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, h1 + h2, fontsize=8, facecolor=C_PANEL, edgecolor=C_GRID)

    ax3 = axes[1]
    style_ax(ax3, 'Amplitude: In-Burst vs Isolated Spikes')
    burst_spike_set = set()
    for b in bursts:
        for i in b['idxs']:
            if i < len(valid_times):
                burst_spike_set.add(i)
    in_burst  = [p2p[i] for i in range(len(p2p)) if i     in burst_spike_set]
    out_burst = [p2p[i] for i in range(len(p2p)) if i not in burst_spike_set]
    data = [d for d in [in_burst, out_burst] if d]
    box_lbls = [lbl for lbl, d in zip(['In-burst', 'Isolated'], [in_burst, out_burst]) if d]
    if data:
        bp = ax3.boxplot(data, tick_labels=box_lbls, patch_artist=True,
                         medianprops=dict(color=C_TEXT, linewidth=2))
        for patch, c in zip(bp['boxes'], [C_BURST, C_SPIKE]):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
        for elem in ['whiskers', 'caps', 'fliers']:
            for item in bp[elem]:
                item.set_color(C_MUTED)
    ax3.set_ylabel('Peak-to-peak amplitude (µV)', fontsize=9)
    fig.tight_layout()
    panels = [
        ('duration_spikes', 'Per-Burst Duration and Spike Count',      [ax, ax2]),
        ('in_burst_vs_iso', 'Amplitude: In-Burst vs Isolated Spikes',  [ax3]),
    ]
    return fig, panels

def plot_amplitude_burst_membership(p2p, in_burst_mask):
    in_burst  = p2p[in_burst_mask]
    out_burst = p2p[~in_burst_mask]
    data     = [d for d in [in_burst, out_burst] if len(d)]
    box_lbls = [lbl for lbl, d in zip(['In-burst', 'Isolated'], [in_burst, out_burst]) if len(d)]

    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor=C_BG)
    style_ax(ax, 'Amplitude: In-Burst vs Isolated Spikes')
    if data:
        bp = ax.boxplot(data, tick_labels=box_lbls, patch_artist=True,
                         medianprops=dict(color=C_TEXT, linewidth=2))
        for patch, c in zip(bp['boxes'], [C_BURST, C_SPIKE]):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
        for elem in ['whiskers', 'caps', 'fliers']:
            for item in bp[elem]:
                item.set_color(C_MUTED)
    ax.set_ylabel('Peak-to-peak amplitude (µV)', fontsize=9)
    fig.tight_layout()
    return fig

def plot_waveform_metrics(widths_ms, p2p, snr_arr):
    valid_w = ~np.isnan(widths_ms)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=C_BG)

    ax1 = axes[0]
    style_ax(ax1, 'Spike Width (Trough-to-Peak)')
    if valid_w.any():
        ax1.hist(widths_ms[valid_w], bins=min(25, max(8, valid_w.sum() // 3)),
                 color=C_SPIKE, alpha=0.75, edgecolor=C_BG)
        ax1.axvline(np.mean(widths_ms[valid_w]), color=C_TEXT, lw=1.2, ls='--',
                    label=f'Mean: {np.mean(widths_ms[valid_w]):.2f} ms')
        ax1.legend(fontsize=8, facecolor=C_PANEL, edgecolor=C_GRID)
    ax1.set_xlabel('Width (ms)', fontsize=9)
    ax1.set_ylabel('Count', fontsize=9)

    ax2 = axes[1]
    style_ax(ax2, 'Amplitude vs Spike Width')
    if valid_w.any():
        ax2.scatter(widths_ms[valid_w], p2p[valid_w], c=C_SPIKE, s=25, alpha=0.6)
    ax2.set_xlabel('Width (ms)', fontsize=9)
    ax2.set_ylabel('Peak-to-peak (µV)', fontsize=9)

    ax3 = axes[2]
    style_ax(ax3, 'SNR Distribution')
    if len(snr_arr):
        ax3.hist(snr_arr, bins=min(25, max(8, len(snr_arr) // 3)),
                 color=C_LOGISI, alpha=0.75, edgecolor=C_BG)
        ax3.axvline(np.mean(snr_arr), color=C_TEXT, lw=1.2, ls='--',
                    label=f'Mean: {np.mean(snr_arr):.1f}×')
        ax3.legend(fontsize=8, facecolor=C_PANEL, edgecolor=C_GRID)
    ax3.set_xlabel('SNR (×σ)', fontsize=9)
    ax3.set_ylabel('Count', fontsize=9)

    fig.tight_layout()
    panels = [
        ('width_hist',    'Spike Width (Trough-to-Peak)', [ax1]),
        ('amp_vs_width',  'Amplitude vs Spike Width',     [ax2]),
        ('snr_hist',      'SNR Distribution',             [ax3]),
    ]
    return fig, panels

def plot_burst_amplitude_dynamics(positions, amps, valid_times, p2p, isi_pre, isi_post):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=C_BG)

    ax1 = axes[0]
    style_ax(ax1, 'Intra-Burst Amplitude Decrement')
    if len(positions):
        ax1.scatter(positions, amps, c=C_MUTED, s=18, alpha=0.35, label='Spikes')
        max_pos = int(positions.max())
        means = [amps[positions == k].mean() for k in range(max_pos + 1)]
        ax1.plot(range(max_pos + 1), means, color=C_BURST, lw=2.0, marker='o',
                 markersize=4, label='Mean per position')
        ax1.legend(fontsize=8, facecolor=C_PANEL, edgecolor=C_GRID)
    ax1.set_xlabel('Spike position in burst', fontsize=9)
    ax1.set_ylabel('Peak-to-peak (µV)', fontsize=9)

    ax2 = axes[1]
    style_ax(ax2, 'Amplitude vs Preceding ISI')
    valid_pre = ~np.isnan(isi_pre)
    if valid_pre.any():
        ax2.scatter(isi_pre[valid_pre], p2p[valid_pre], c=C_SPIKE, s=25, alpha=0.6)
    ax2.set_xlabel('Preceding ISI (ms)', fontsize=9)
    ax2.set_ylabel('Peak-to-peak (µV)', fontsize=9)

    ax3 = axes[2]
    style_ax(ax3, 'Amplitude vs Following ISI')
    valid_post = ~np.isnan(isi_post)
    if valid_post.any():
        ax3.scatter(isi_post[valid_post], p2p[valid_post], c=C_LOGISI, s=25, alpha=0.6)
    ax3.set_xlabel('Following ISI (ms)', fontsize=9)
    ax3.set_ylabel('Peak-to-peak (µV)', fontsize=9)

    fig.tight_layout()
    panels = [
        ('intraburst_decrement', 'Intra-Burst Amplitude Decrement', [ax1]),
        ('amp_vs_pre_isi',       'Amplitude vs Preceding ISI',      [ax2]),
        ('amp_vs_post_isi',      'Amplitude vs Following ISI',      [ax3]),
    ]
    return fig, panels

def _scatter_with_fit(ax, x, y, title, xlabel, ylabel, color):
    style_ax(ax, title)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    ax.scatter(x, y, c=color, s=25, alpha=0.6)
    if len(x) >= 3:
        r = np.corrcoef(x, y)[0, 1]
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, color=C_ANNOT, lw=1.4,
                label=f'r = {r:.2f}')
        ax.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

def plot_burst_correlations(burst_stats):
    if not burst_stats:
        return None, []
    durs  = np.array([b['duration'] for b in burst_stats], dtype=float)
    ns    = np.array([b['n_spikes'] for b in burst_stats], dtype=float)
    mamp  = np.array([b['mean_amp'] for b in burst_stats], dtype=float)
    atten = np.array([b['attenuation_index'] for b in burst_stats], dtype=float)
    mwid  = np.array([b['mean_width'] for b in burst_stats], dtype=float)

    fig = plt.figure(figsize=(15, 8), facecolor=C_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle('Burst-Level Correlations', color=C_TEXT, fontsize=13, fontweight='bold')

    ax1 = fig.add_subplot(gs[0, 0])
    _scatter_with_fit(ax1, durs, mamp,
                       'Duration vs Mean Amplitude', 'Duration (ms)', 'Mean P2P (µV)', C_SPIKE)
    ax2 = fig.add_subplot(gs[0, 1])
    _scatter_with_fit(ax2, durs, atten,
                       'Duration vs Attenuation Index', 'Duration (ms)', 'Attenuation Index', C_BURST)
    ax3 = fig.add_subplot(gs[0, 2])
    _scatter_with_fit(ax3, ns, atten,
                       'Spike Count vs Attenuation Index', 'Spike count', 'Attenuation Index', C_LOGISI)
    ax4 = fig.add_subplot(gs[1, 0])
    _scatter_with_fit(ax4, durs, mwid,
                       'Duration vs Mean Width', 'Duration (ms)', 'Mean Width (ms)', C_SPIKE)
    ax5 = fig.add_subplot(gs[1, 1])
    _scatter_with_fit(ax5, durs, ns,
                       'Duration vs Spike Count', 'Duration (ms)', 'Spike count', C_MUTED)

    panels = [
        ('duration_vs_mean_amp',      'Duration vs Mean Amplitude',        [ax1]),
        ('duration_vs_attenuation',   'Duration vs Attenuation Index',     [ax2]),
        ('spikecount_vs_attenuation', 'Spike Count vs Attenuation Index',  [ax3]),
        ('duration_vs_width',         'Duration vs Mean Width',            [ax4]),
        ('duration_vs_spikecount',    'Duration vs Spike Count',           [ax5]),
    ]
    return fig, panels

def plot_logisi_histogram(hist_data, isi_th_ms, void_param, fallback):
    if hist_data is None or len(hist_data['bin_centers']) == 0:
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=C_BG)
        style_ax(ax, 'logISI Histogram — insufficient ISIs for analysis')
        return fig

    bc       = hist_data['bin_centers']
    counts   = hist_data['counts']
    smoothed = hist_data['smoothed']
    p1       = hist_data.get('p1')
    p2       = hist_data.get('p2')

    isi_vals = 10 ** bc
    log_th   = np.log10(max(isi_th_ms, 1e-3))
    bar_clrs = [C_SPIKE if b < log_th else C_MUTED for b in bc]

    if len(isi_vals) > 1:
        widths = np.diff(np.append(isi_vals, isi_vals[-1] * 10 ** 0.1)) * 0.80
    else:
        widths = np.array([isi_vals[0] * 0.5])

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=C_BG)
    style_ax(ax, 'logISI Histogram (Pasquale et al. 2010)')

    for x, cnt, w, c in zip(isi_vals, counts, widths, bar_clrs):
        ax.bar(x, cnt, width=w, color=c, edgecolor='none', alpha=0.65, align='center')

    ax.plot(isi_vals, smoothed, color=C_TEXT, linewidth=1.8, zorder=5, alpha=0.85,
            label='Gaussian-smoothed')

    if p1 is not None and p1 < len(isi_vals):
        ax.axvline(isi_vals[p1], color=C_SPIKE, ls=':', lw=1.4, alpha=0.8,
                   label=f'Intra-burst peak ({isi_vals[p1]:.1f} ms)')
    if p2 is not None and p2 < len(isi_vals):
        ax.axvline(isi_vals[p2], color=C_MUTED, ls=':', lw=1.4, alpha=0.8,
                   label=f'Inter-burst peak ({isi_vals[p2]:.1f} ms)')

    thresh_lbl = (f'ISIth = {isi_th_ms:.1f} ms  [fallback: 100 ms]' if fallback
                  else f'ISIth = {isi_th_ms:.1f} ms  (void = {void_param:.3f})')
    ax.axvline(isi_th_ms, color=C_BURST, lw=2.0, ls='--', zorder=6, label=thresh_lbl)

    ax.set_xscale('log')
    ax.set_xlabel('ISI (ms, log scale)', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.legend(fontsize=8, facecolor=C_PANEL, edgecolor=C_GRID)

    note = ('Fallback to 100 ms — no bimodal structure detected' if fallback
            else f'Bimodal structure detected (void = {void_param:.3f})')
    ax.text(0.98, 0.97, note, transform=ax.transAxes, ha='right', va='top',
            color=C_MUTED, fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=C_PANEL, edgecolor=C_GRID))
    fig.tight_layout()
    return fig

def plot_comparison_raster(spike_times, bursts_mi, bursts_logisi, hamming_pct, agreement):
    agreement_clean = _strip_emoji(agreement)
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), facecolor=C_BG, sharex=True)
    fig.suptitle(
        f'Burst Detection Comparison — {agreement_clean}  (Hamming = {hamming_pct:.1f}%)',
        color=C_TEXT, fontsize=11, fontweight='bold'
    )
    for ax, bursts, color, title in [
        (axes[0], bursts_mi,     C_BURST,  'Max Interval (Cotterill et al. 2016)'),
        (axes[1], bursts_logisi, C_LOGISI, 'logISI Adaptive (Pasquale et al. 2010)'),
    ]:
        style_ax(ax, title)
        for b in bursts:
            ax.axvspan(b['start'], b['end'], color=color, alpha=0.25)
        ax.eventplot(spike_times, lineoffsets=0, linelengths=0.7,
                     color=C_TEXT, linewidths=0.9, alpha=0.50)
        ax.set_yticks([])
        ax.set_ylabel('Spikes', fontsize=8)
        if len(spike_times) > 0:
            ax.set_xlim(spike_times[0] - 0.3, spike_times[-1] + 0.3)
    axes[-1].set_xlabel('Time (s)', fontsize=9)
    fig.tight_layout()
    panels = [
        ('mi_raster',     'Max Interval Burst Epochs',      [axes[0]]),
        ('logisi_raster', 'logISI Adaptive Burst Epochs',   [axes[1]]),
    ]
    return fig, panels

def plot_raw_trace(t, v, label, color, max_points=50_000):
    """Plot a continuous voltage trace over time, decimating if very large."""
    n = len(t)
    if n > max_points:
        step = int(np.ceil(n / max_points))
        t_plot, v_plot = t[::step], v[::step]
        decim_note = f" (downsampled {step}×, {len(t_plot):,} of {n:,} pts)"
    else:
        t_plot, v_plot = t, v
        decim_note = ""

    fig, ax = plt.subplots(figsize=(14, 4), facecolor=C_BG)
    style_ax(ax, f'{label} Voltage Trace{decim_note}')
    ax.plot(t_plot, v_plot, color=color, linewidth=0.5)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Voltage (µV)', fontsize=9)
    ax.set_xlim(t_plot[0], t_plot[-1])
    fig.tight_layout()
    return fig

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
        "Raw voltage trace (.txt or .csv) — multiple segment files allowed",
        type=['txt', 'csv'],
        accept_multiple_files=True,
        help="Two-column file: time (s) and voltage (µV), tab- or comma-separated. "
             "Upload multiple sequential export segments to auto-stitch them into one trace."
    )
with col2:
    ne_file = st.file_uploader(
        "NeuroExplorer export (.txt) — optional",
        type=['txt'],
        help="Overlays NeuroExplorer spike timestamps instead of detecting from the raw signal."
    )

if not raw_files:
    st.info(
        "Upload a raw MEA data file to begin. "
        "The file should have two columns: **time (s)** and **voltage (µV)**, "
        "exported from Multi Channel Analyzer as *Raw Data Time Points*. "
        "Upload multiple segment files at once if your recording was exported in chunks."
    )
    st.stop()

# ── Parse & filter ────────────────────────────────────────────────────────────
with st.spinner("Loading and filtering signal..."):
    segments = [parse_raw_file(f.read()) for f in raw_files]
    stitch_summary = None
    if len(segments) == 1:
        raw_t, raw_v = segments[0]
    else:
        raw_t, raw_v, stitch_summary = stitch_segments(segments)
    fs       = round(1.0 / np.median(np.diff(raw_t)))
    raw_filt = bandpass_filter(raw_v, fs, bp_low, bp_high)

if stitch_summary is not None:
    with st.expander(
        f"Stitched {len(raw_files)} segments into one "
        f"{raw_t[-1] - raw_t[0]:.1f}s trace", expanded=False
    ):
        st.dataframe(pd.DataFrame([{
            'Order':                      k + 1,
            'File':                       raw_files[row['segment_index']].name,
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

ne_spike_times, ne_isis, ne_freqs = None, None, None
if ne_file:
    ne_spike_times, ne_isis, ne_freqs = parse_neuroexplorer_file(ne_file.read())

analysis_spikes = ne_spike_times if ne_spike_times is not None else spike_times
analysis_freqs  = ne_freqs if ne_freqs is not None else np.full(len(spike_times), np.nan)
rec_dur = float(raw_t[-1] - raw_t[0])

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

with st.spinner("Extracting waveforms..."):
    waveforms, troughs, peaks, p2p, valid_times, t_axis = extract_waveforms(
        raw_t, raw_filt, analysis_spikes, fs, pre_ms, post_ms
    )

widths_ms = compute_spike_widths(waveforms, t_axis) if len(waveforms) else np.array([])
isi_pre, isi_post = compute_isi_arrays(valid_times)
snr_arr = np.abs(troughs) / noise_floor if len(troughs) else np.array([])
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
mean_fr   = n_spikes / rec_dur
n_bursts  = len(bursts)
pct_burst = 100 * sum(b['n_spikes'] for b in bursts) / max(n_spikes, 1)
mean_p2p  = float(p2p.mean()) if len(p2p) > 0 else 0.0
snr       = abs(float(troughs.mean())) / noise_floor if len(troughs) > 0 else 0.0

base_metrics = [
    ("Total Spikes",     str(n_spikes)),
    ("Recording (s)",    f"{rec_dur:.2f}"),
    ("Mean Firing Rate", f"{mean_fr:.2f} Hz"),
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

metric_cols = st.columns(len(base_metrics))
for col, (label, value) in zip(metric_cols, base_metrics):
    col.metric(label, value)

if ne_spike_times is not None:
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
tab1, tab_raw, tab2, tab3, tab_wave, tab4, tab_dyn, tab5, tab6 = st.tabs([
    "Overview", "Raw Trace", "ISI Analysis", "Amplitude", "Waveform Metrics",
    "Bursts", "Burst Amplitude Dynamics", "logISI Histogram", "Data Table"
])

with tab1:
    fig_ov, panels_ov = plot_overview(analysis_spikes, bursts, analysis_freqs)
    st.pyplot(fig_ov, use_container_width=True)
    st.download_button("Download PNG (combined)", fig_to_bytes(fig_ov), "overview.png", "image/png")
    render_panel_downloads(fig_ov, panels_ov, "overview")
    plt.close(fig_ov)

with tab_raw:
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
    st.pyplot(fig_raw, use_container_width=True)
    st.download_button("Download PNG", fig_to_bytes(fig_raw), fname, "image/png")
    plt.close(fig_raw)

with tab2:
    if len(analysis_spikes) > 1:
        fig_isi, panels_isi = plot_isi(analysis_spikes, isi_disp, sec_disp)
        st.pyplot(fig_isi, use_container_width=True)
        st.download_button("Download PNG (combined)", fig_to_bytes(fig_isi), "isi_analysis.png", "image/png")
        render_panel_downloads(fig_isi, panels_isi, "isi_analysis")
        isis_all = np.diff(analysis_spikes) * 1000
        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("Mean ISI", f"{isis_all.mean():.1f} ms")
        ic2.metric("Intra-burst ISI",
                   f"{isis_all[isis_all <= isi_disp].mean():.1f} ms" if any(isis_all <= isi_disp) else "—")
        ic3.metric("Inter-burst ISI",
                   f"{isis_all[isis_all > isi_disp].mean():.0f} ms"  if any(isis_all > isi_disp)  else "—")
        plt.close(fig_isi)

with tab3:
    if len(waveforms) > 0:
        fig_amp, panels_amp = plot_amplitude(waveforms, troughs, peaks, p2p,
                                  valid_times, t_axis, noise_floor, threshold)
        st.pyplot(fig_amp, use_container_width=True)
        st.download_button("Download PNG (combined)", fig_to_bytes(fig_amp), "amplitude.png", "image/png")
        render_panel_downloads(fig_amp, panels_amp, "amplitude")
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("Mean Trough", f"{troughs.mean():.2f} µV")
        ac2.metric("Mean P2P",    f"{p2p.mean():.2f} µV")
        ac3.metric("P2P Std Dev", f"{p2p.std():.2f} µV")
        ac4.metric("SNR",         f"{snr:.1f}×")
        plt.close(fig_amp)

        if in_burst_mask.any() and (~in_burst_mask).any():
            fig_ib = plot_amplitude_burst_membership(p2p, in_burst_mask)
            st.pyplot(fig_ib, use_container_width=True)
            st.download_button("Download PNG", fig_to_bytes(fig_ib),
                               "amplitude_burst_membership.png", "image/png")
            plt.close(fig_ib)

with tab_wave:
    if len(widths_ms) > 0:
        fig_wm, panels_wm = plot_waveform_metrics(widths_ms, p2p, snr_arr)
        st.pyplot(fig_wm, use_container_width=True)
        st.download_button("Download PNG (combined)", fig_to_bytes(fig_wm), "waveform_metrics.png", "image/png")
        render_panel_downloads(fig_wm, panels_wm, "waveform_metrics")
        plt.close(fig_wm)
    else:
        st.info("No waveforms available to compute width/SNR metrics.")

with tab4:
    if bursts:
        fig_b, panels_b = plot_bursts(analysis_spikes, bursts, p2p, valid_times)
        if fig_b:
            st.pyplot(fig_b, use_container_width=True)
            st.download_button("Download PNG (combined)", fig_to_bytes(fig_b), "bursts.png", "image/png")
            render_panel_downloads(fig_b, panels_b, "bursts")
            plt.close(fig_b)
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
            st.pyplot(fig_bc, use_container_width=True)
            st.download_button("Download PNG (combined)", fig_to_bytes(fig_bc), "burst_correlations.png", "image/png")
            render_panel_downloads(fig_bc, panels_bc, "burst_correlations")
            plt.close(fig_bc)
    else:
        st.warning("No bursts were detected with the current parameters.")

with tab_dyn:
    if len(decr_positions) > 0 or (~np.isnan(isi_pre)).any():
        fig_dyn, panels_dyn = plot_burst_amplitude_dynamics(decr_positions, decr_amps, valid_times, p2p, isi_pre, isi_post)
        st.pyplot(fig_dyn, use_container_width=True)
        st.download_button("Download PNG (combined)", fig_to_bytes(fig_dyn),
                           "burst_amplitude_dynamics.png", "image/png")
        render_panel_downloads(fig_dyn, panels_dyn, "burst_amplitude_dynamics")
        plt.close(fig_dyn)
    else:
        st.info("Not enough multi-spike bursts / ISIs to compute amplitude dynamics.")

with tab5:
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

    st.pyplot(fig_logi, use_container_width=True)
    st.download_button("Download PNG", fig_to_bytes(fig_logi), "logisi_histogram.png", "image/png")
    plt.close(fig_logi)

    if burst_method == "Both — compare methods":
        st.markdown("#### Burst Epoch Comparison Raster")
        fig_cmp, panels_cmp = plot_comparison_raster(
            analysis_spikes, bursts_mi, bursts_logisi, hamming_pct, agreement
        )
        st.pyplot(fig_cmp, use_container_width=True)
        st.download_button("Download PNG (combined)", fig_to_bytes(fig_cmp),
                           "comparison_raster.png", "image/png")
        render_panel_downloads(fig_cmp, panels_cmp, "comparison_raster")
        plt.close(fig_cmp)

with tab6:
    if len(valid_times) > 0:
        df = build_summary_df(analysis_spikes, bursts, troughs, p2p, valid_times, noise_floor)
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

    methods_text = (
        f"Spikes were detected as negative threshold crossings exceeding {thr_mult}× "
        f"the estimated noise floor, computed as the median absolute deviation of the "
        f"bandpass-filtered signal ({int(bp_low)}–{int(bp_high)} Hz, 4th-order Butterworth; "
        f"Quiroga et al. 2004), with a 1 ms refractory period. "
        f"Bursts were identified using the {method_name} method {citation}. "
        f"{method_frag}{compare_frag} "
        f"Spike amplitude was quantified as peak-to-peak voltage of the filtered waveform "
        f"within a {pre_ms} ms–{post_ms} ms window around each threshold crossing "
        f"(Obien et al. 2015)."
    )
    st.text_area("Methods section paragraph", methods_text, height=200)

st.markdown("---")
st.caption(
    "MEA Spike Analyser — "
    "Cotterill et al. (2016) J Neurophysiol · "
    "Pasquale et al. (2010) J Comput Neurosci"
)
