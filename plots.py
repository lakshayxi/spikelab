from io import BytesIO

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.transforms import Bbox

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

MAX_WAVEFORM_OVERLAYS = 500
MAX_SCATTER_POINTS = 5_000


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


def select_display_indices(length, max_items):
    """Return deterministic, approximately even indices for display-only sampling."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if max_items <= 0:
        raise ValueError("max_items must be positive")
    if length <= max_items:
        return np.arange(length, dtype=int)
    if max_items == 1:
        return np.array([0, length - 1], dtype=int)
    return np.unique(np.linspace(0, length - 1, max_items, dtype=int))


def _display_note(displayed, total, representative_label, analysed_label, context=None):
    if displayed >= total:
        return None
    note = f"Showing {displayed:,} representative {representative_label} from {total:,} analysed {analysed_label}"
    if context:
        note += f" in {context}"
    return note + "."


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
    """Render descriptive per-panel download buttons inside a figure popover."""
    if not panels:
        return
    for short_name, label, axes in panels:
        display_label = label
        if " — " in display_label and len(display_label.split(" — ", 1)[0]) == 1:
            display_label = display_label.split(" — ", 1)[1]
        st.download_button(
            f"Download {display_label.lower()} PNG",
            panel_to_bytes(fig, axes),
            f"{base_filename}_{short_name}.png",
            "image/png",
            key=f"dl_{base_filename}_{short_name}",
            width="stretch",
        )

def plot_overview(spike_times, bursts, freqs, x_range=None):
    if x_range is None:
        x_min, x_max = spike_times[0] - 0.3, spike_times[-1] + 0.3
    else:
        x_min, x_max = x_range
    visible_spikes = spike_times[(spike_times >= x_min) & (spike_times <= x_max)]
    visible_bursts = [(k, b) for k, b in enumerate(bursts) if b['end'] >= x_min and b['start'] <= x_max]

    fig = plt.figure(figsize=(14, 6), facecolor=C_BG)
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45,
                            left=0.06, right=0.98, top=0.90, bottom=0.10)
    fig.suptitle('Spike Raster and Firing Rate', color=C_TEXT, fontsize=13, fontweight='bold')

    ax1 = fig.add_subplot(gs[0])
    style_ax(ax1, 'Spike Raster')
    for _, b in visible_bursts:
        ax1.axvspan(b['start'], b['end'], color=C_BURST, alpha=0.15)
    if len(visible_spikes):
        ax1.eventplot(visible_spikes, lineoffsets=0, linelengths=0.7,
                      color=C_SPIKE, linewidths=1.2)
    for k, b in visible_bursts:
        mid = (b['start'] + b['end']) / 2
        ax1.text(mid, 0.55, f"B{k+1}", color=C_BURST, fontsize=7, ha='center',
                 fontweight='bold', transform=ax1.get_xaxis_transform())
    ax1.set_yticks([])
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylabel('Spikes', fontsize=9)

    ax2 = fig.add_subplot(gs[1])
    style_ax(ax2, 'Instantaneous Firing Rate')
    freq_vals = freqs[:len(spike_times)]
    valid = (~np.isnan(freq_vals)) & (spike_times >= x_min) & (spike_times <= x_max)
    if valid.any():
        ax2.fill_between(spike_times[valid], freq_vals[valid],
                         alpha=0.2, color=C_SPIKE)
        ax2.plot(spike_times[valid], freq_vals[valid],
                 color=C_SPIKE, linewidth=0.9)
    for _, b in visible_bursts:
        ax2.axvspan(b['start'], b['end'], color=C_BURST, alpha=0.10)
    ax2.set_xlabel('Time (s)', fontsize=9)
    ax2.set_ylabel('Freq (Hz)', fontsize=9)
    ax2.set_xlim(x_min, x_max)
    panels = [
        ('raster',       'Spike Raster',               [ax1]),
        ('firing_rate',  'Instantaneous Firing Rate',  [ax2]),
    ]
    return fig, panels

def plot_isi(spike_times, isi_threshold_ms, secondary_threshold_ms=None):
    isis = np.diff(spike_times) * 1000
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=C_BG)

    bins    = np.linspace(0, min(isis.max(), 1000), 60)
    centers = (bins[:-1] + bins[1:]) / 2
    counts, _ = np.histogram(isis, bins=bins)
    colors  = [C_SPIKE if c <= isi_threshold_ms else C_MUTED for c in centers]

    for ax, ylabel, yscale in [(axes[0], 'Count', 'linear'), (axes[1], 'Count (log)', 'log')]:
        title = 'Linear scale' if yscale == 'linear' else 'Log count scale'
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
        ('linear', 'Linear-scale ISI histogram', [axes[0]]),
        ('log',    'Log-count ISI histogram',    [axes[1]]),
    ]
    return fig, panels

def plot_amplitude(
    waveforms,
    troughs,
    peaks,
    p2p,
    valid_times,
    t_axis,
    noise_floor,
    threshold,
    max_waveforms=MAX_WAVEFORM_OVERLAYS,
    max_scatter_points=MAX_SCATTER_POINTS,
):
    display_notes = []
    fig = plt.figure(figsize=(14, 10), facecolor=C_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32,
                            left=0.07, right=0.97, top=0.92, bottom=0.07)
    fig.suptitle('Spike Amplitude Quantification', color=C_TEXT, fontsize=13, fontweight='bold')
    cmap     = plt.cm.viridis
    norm_amp = ((np.abs(troughs) - np.abs(troughs).min()) /
                (np.abs(troughs).max() - np.abs(troughs).min() + 1e-9))
    waveform_indices = select_display_indices(len(waveforms), max_waveforms)
    waveform_note = _display_note(
        len(waveform_indices),
        len(waveforms),
        "waveforms",
        "waveforms",
    )
    if waveform_note:
        display_notes.append(waveform_note)

    ax_A = fig.add_subplot(gs[0, 0])
    style_ax(ax_A, 'A  All Spike Waveforms')
    for idx in waveform_indices:
        ax_A.plot(t_axis, waveforms[idx], color=cmap(norm_amp[idx]), alpha=0.20, linewidth=0.5)
    mean_w = waveforms.mean(axis=0)
    ax_A.plot(t_axis, mean_w, color=C_TEXT, linewidth=2.0, label='Mean waveform', zorder=5)
    if np.isfinite(threshold):
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
    if np.isfinite(threshold):
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
    scatter_indices = select_display_indices(len(valid_times), max_scatter_points)
    scatter_note = _display_note(
        len(scatter_indices),
        len(valid_times),
        "points",
        "spikes",
        "Spike Amplitude Over Time",
    )
    if scatter_note:
        display_notes.append(scatter_note)
    p2p_norm = plt.Normalize(p2p.min(), p2p.max())
    sc = ax_D.scatter(
        valid_times[scatter_indices],
        p2p[scatter_indices],
        c=p2p[scatter_indices],
        cmap=cmap,
        norm=p2p_norm,
        s=35,
        alpha=0.80,
        zorder=3,
    )
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
    return fig, panels, display_notes

def plot_bursts(bursts, p2p, in_burst_mask):
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
    in_burst  = p2p[in_burst_mask]
    out_burst = p2p[~in_burst_mask]
    data = [d for d in [in_burst, out_burst] if len(d)]
    box_lbls = [lbl for lbl, d in zip(['In-burst', 'Isolated'], [in_burst, out_burst]) if len(d)]
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

def plot_waveform_metrics(widths_ms, p2p, snr_arr, max_scatter_points=MAX_SCATTER_POINTS):
    display_notes = []
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
        valid_indices = np.flatnonzero(valid_w)
        selected = select_display_indices(len(valid_indices), max_scatter_points)
        scatter_indices = valid_indices[selected]
        ax2.scatter(widths_ms[scatter_indices], p2p[scatter_indices], c=C_SPIKE, s=25, alpha=0.6)
        scatter_note = _display_note(
            len(scatter_indices),
            len(valid_indices),
            "points",
            "spikes",
            "Amplitude vs Spike Width",
        )
        if scatter_note:
            display_notes.append(scatter_note)
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
    return fig, panels, display_notes


def plot_burst_amplitude_dynamics(
    positions,
    amps,
    valid_times,
    p2p,
    isi_pre,
    isi_post,
    max_scatter_points=MAX_SCATTER_POINTS,
):
    display_notes = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=C_BG)

    ax1 = axes[0]
    style_ax(ax1, 'Intra-Burst Amplitude Decrement')
    if len(positions):
        selected = select_display_indices(len(positions), max_scatter_points)
        ax1.scatter(positions[selected], amps[selected], c=C_MUTED, s=18, alpha=0.35, label='Spikes')
        max_pos = int(positions.max())
        means = [amps[positions == k].mean() for k in range(max_pos + 1)]
        ax1.plot(range(max_pos + 1), means, color=C_BURST, lw=2.0, marker='o',
                 markersize=4, label='Mean per position')
        ax1.legend(fontsize=8, facecolor=C_PANEL, edgecolor=C_GRID)
        scatter_note = _display_note(
            len(selected),
            len(positions),
            "points",
            "in-burst spikes",
            "Intra-Burst Amplitude Decrement",
        )
        if scatter_note:
            display_notes.append(scatter_note)
    ax1.set_xlabel('Spike position in burst', fontsize=9)
    ax1.set_ylabel('Peak-to-peak (µV)', fontsize=9)

    ax2 = axes[1]
    style_ax(ax2, 'Amplitude vs Preceding ISI')
    valid_pre = ~np.isnan(isi_pre)
    if valid_pre.any():
        valid_indices = np.flatnonzero(valid_pre)
        selected = select_display_indices(len(valid_indices), max_scatter_points)
        scatter_indices = valid_indices[selected]
        ax2.scatter(isi_pre[scatter_indices], p2p[scatter_indices], c=C_SPIKE, s=25, alpha=0.6)
        scatter_note = _display_note(
            len(scatter_indices),
            len(valid_indices),
            "points",
            "spikes",
            "Amplitude vs Preceding ISI",
        )
        if scatter_note:
            display_notes.append(scatter_note)
    ax2.set_xlabel('Preceding ISI (ms)', fontsize=9)
    ax2.set_ylabel('Peak-to-peak (µV)', fontsize=9)

    ax3 = axes[2]
    style_ax(ax3, 'Amplitude vs Following ISI')
    valid_post = ~np.isnan(isi_post)
    if valid_post.any():
        valid_indices = np.flatnonzero(valid_post)
        selected = select_display_indices(len(valid_indices), max_scatter_points)
        scatter_indices = valid_indices[selected]
        ax3.scatter(isi_post[scatter_indices], p2p[scatter_indices], c=C_LOGISI, s=25, alpha=0.6)
        scatter_note = _display_note(
            len(scatter_indices),
            len(valid_indices),
            "points",
            "spikes",
            "Amplitude vs Following ISI",
        )
        if scatter_note:
            display_notes.append(scatter_note)
    ax3.set_xlabel('Following ISI (ms)', fontsize=9)
    ax3.set_ylabel('Peak-to-peak (µV)', fontsize=9)

    fig.tight_layout()
    panels = [
        ('intraburst_decrement', 'Intra-Burst Amplitude Decrement', [ax1]),
        ('amp_vs_pre_isi',       'Amplitude vs Preceding ISI',      [ax2]),
        ('amp_vs_post_isi',      'Amplitude vs Following ISI',      [ax3]),
    ]
    return fig, panels, display_notes


def _scatter_with_fit(ax, x, y, title, xlabel, ylabel, color, display_notes, max_scatter_points):
    style_ax(ax, title)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    selected = select_display_indices(len(x), max_scatter_points)
    ax.scatter(x[selected], y[selected], c=color, s=25, alpha=0.6)
    scatter_note = _display_note(len(selected), len(x), "points", "bursts", title)
    if scatter_note:
        display_notes.append(scatter_note)
    if len(x) >= 3:
        r = np.corrcoef(x, y)[0, 1]
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, color=C_ANNOT, lw=1.4,
                label=f'r = {r:.2f}')
        ax.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

def plot_burst_correlations(burst_stats, max_scatter_points=MAX_SCATTER_POINTS):
    if not burst_stats:
        return None, [], []
    display_notes = []
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
                      'Duration vs Mean Amplitude', 'Duration (ms)', 'Mean P2P (µV)', C_SPIKE,
                      display_notes, max_scatter_points)
    ax2 = fig.add_subplot(gs[0, 1])
    _scatter_with_fit(ax2, durs, atten,
                      'Duration vs Attenuation Index', 'Duration (ms)', 'Attenuation Index', C_BURST,
                      display_notes, max_scatter_points)
    ax3 = fig.add_subplot(gs[0, 2])
    _scatter_with_fit(ax3, ns, atten,
                      'Spike Count vs Attenuation Index', 'Spike count', 'Attenuation Index', C_LOGISI,
                      display_notes, max_scatter_points)
    ax4 = fig.add_subplot(gs[1, 0])
    _scatter_with_fit(ax4, durs, mwid,
                      'Duration vs Mean Width', 'Duration (ms)', 'Mean Width (ms)', C_SPIKE,
                      display_notes, max_scatter_points)
    ax5 = fig.add_subplot(gs[1, 1])
    _scatter_with_fit(ax5, durs, ns,
                      'Duration vs Spike Count', 'Duration (ms)', 'Spike count', C_MUTED,
                      display_notes, max_scatter_points)

    panels = [
        ('duration_vs_mean_amp',      'Duration vs Mean Amplitude',        [ax1]),
        ('duration_vs_attenuation',   'Duration vs Attenuation Index',     [ax2]),
        ('spikecount_vs_attenuation', 'Spike Count vs Attenuation Index',  [ax3]),
        ('duration_vs_width',         'Duration vs Mean Width',            [ax4]),
        ('duration_vs_spikecount',    'Duration vs Spike Count',           [ax5]),
    ]
    return fig, panels, display_notes

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
        f'Burst Detection Comparison — {agreement_clean}  '
        f'(Hamming distance = {hamming_pct:.1f}%)',
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
    display_notes = []
    if n > max_points:
        step = int(np.ceil(n / max_points))
        t_plot, v_plot = t[::step], v[::step]
        display_notes.append(
            f"Displayed using {step}× visual downsampling: {len(t_plot):,} of "
            f"{n:,} samples. Analysis uses the full signal."
        )
    else:
        t_plot, v_plot = t, v

    fig, ax = plt.subplots(figsize=(14, 4), facecolor=C_BG)
    style_ax(ax, f'{label} voltage trace')
    ax.plot(t_plot, v_plot, color=color, linewidth=0.5)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Voltage (µV)', fontsize=9)
    ax.set_xlim(t_plot[0], t_plot[-1])
    fig.tight_layout()
    return fig, display_notes


def plot_spike_waveforms(waveforms, t_axis, max_waveforms=500):
    """Plot NeuroExplorer-supplied spike snippets and their all-spike mean.

    A deterministic, evenly spaced subset keeps the figure responsive for
    channels with many spikes; the mean and standard deviation still use every
    supplied waveform.
    """
    n_waveforms = len(waveforms)
    if n_waveforms > max_waveforms:
        shown_indices = select_display_indices(n_waveforms, max_waveforms)
        shown = waveforms[shown_indices]
        display_notes = [
            f"Displayed using {len(shown):,} representative waveforms from "
            f"{n_waveforms:,} supplied waveforms. Mean and variability use all waveforms."
        ]
    else:
        shown = waveforms
        display_notes = []

    mean_waveform = waveforms.mean(axis=0)
    std_waveform = waveforms.std(axis=0)

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=C_BG)
    style_ax(ax, 'Representative spike waveforms')
    ax.plot(t_axis, shown.T, color=C_SPIKE, alpha=0.08, linewidth=0.45)
    ax.fill_between(
        t_axis,
        mean_waveform - std_waveform,
        mean_waveform + std_waveform,
        color=C_MUTED,
        alpha=0.18,
        label='Mean ± 1 SD (all spikes)',
    )
    ax.plot(t_axis, mean_waveform, color=C_TEXT, linewidth=2.2, label='Mean (all spikes)', zorder=5)
    ax.axvline(0, color=C_MUTED, linewidth=0.8, linestyle=':')
    ax.axhline(0, color=C_GRID, linewidth=0.8)
    ax.set_xlabel('Time from spike timestamp (ms)', fontsize=9)
    ax.set_ylabel('Voltage (µV)', fontsize=9)
    ax.set_xlim(t_axis[0], t_axis[-1])
    ax.legend(fontsize=8, facecolor=C_PANEL, edgecolor=C_GRID)
    fig.tight_layout()
    return fig, display_notes
