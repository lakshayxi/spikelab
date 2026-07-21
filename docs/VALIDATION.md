# Scientific Regression Validation

MEA Spike Analyser uses compact synthetic fixtures to detect implementation
regressions in its burst algorithms, filtered spike-detection pipeline,
waveform edge handling, and EDF calibration. The fixtures are deterministic,
inline in the test suite, and small enough to inspect directly.

## Deterministic fixtures

| Fixture | Synthetic input | Expected observable outputs |
| --- | --- | --- |
| Max Interval: one burst | Spike times `[0, .05, .10, .15, .50]` seconds | One burst from `0` to `.15` seconds; `150 ms`; four spikes; indices `0–3`. |
| Max Interval: two bursts | Spike times `[0, .05, .10, .60, .65, .70]` seconds | Two three-spike bursts at `0–.10` and `.60–.70` seconds; indices `0–2` and `3–5`. |
| Max Interval: merge rule | Spike times `[0, .05, .10, .25, .30, .35]` seconds; beginning/end ISI `100 ms`; minimum IBI `250 ms` | The two candidates merge into one six-spike burst from `0` to `.35` seconds with indices `0–5`. |
| Max Interval: minimum spikes | Two spikes with `min_spikes=3` | No burst. |
| Max Interval: minimum duration | Three spikes spanning `4 ms` with a `10 ms` minimum duration | No burst. |
| Max Interval: time-origin invariance | The one-burst train repeated after a `17.25 s` offset | Duration, count, and indices are unchanged; start and end each shift by `17.25 s`. |
| logISI: clear bimodal train | Five groups of eight spikes using explicit `15–28 ms` intra-group ISIs, `800 ms` gaps, and a distant sentinel | No fallback; threshold `10^1.85 = 70.79457843841391 ms`; void parameter `1.0`; five exact eight-index bursts; sentinel excluded. |
| logISI: uniform train | Sixty `200 ms` ISIs | Fallback enabled; threshold exactly `100 ms`; void parameter `0.0`; no bursts. |
| logISI: dual threshold | Twelve eight-spike cores using explicit `35–58 ms` ISIs and `2 s` gaps; the first core has `120 ms` flanks; a distant sentinel follows | No fallback; threshold `10^2.45 = 281.83829312644605 ms`; void parameter `1.0`; twelve bursts; both first-core flank spikes included; sentinel excluded. |
| Filtered spike pipeline | `20 kHz`, 4,000 samples, seed `2026`, `5 µV` Gaussian noise, `-200 µV` impulses at samples `10`, `40`, `50`, `1000`, `3950`, and `3980`; `300–3000 Hz` fourth-order filter; `5×` threshold; fixed `1 ms` refractory period | Six threshold crossings and exactly five retained detections. The sample-`50` crossing is refractory-suppressed. Retained crossings match samples `10`, `40`, `1000`, `3950`, and `3980` within two samples (`100 µs`), with no additional detections, and a repeated run is identical. |
| Waveform boundaries | Default `1 ms` pre-window and `2 ms` post-window applied to the filtered detections | Three complete waveforms remain, corresponding to injected samples `40`, `1000`, and `3950`. The first detection lacks a complete pre-window and the last lacks a complete post-window. |
| EDF calibration | Two named channels; select channel two; two records; two samples per `0.5 s` record; digital `-100..100` mapped to physical `-2..2`; dimensions `V`, `mV`, and `uV` | Selected label is preserved; declared rate is exactly `4 Hz`; timestamps are `[0, .25, .50, .75]`; voltages are exactly `[-2, -1, 1, 2]` times `1,000,000`, `1,000`, or `1` in internal µV units. |
| EDF unsupported unit | Physical dimension `Ohm` | Parsing raises an explicit unsupported-dimension error. |

The exact `10^1.85` and `10^2.45` logISI thresholds are implementation
regression checks for the current histogram binning, smoothing, and valley
selection. They are not claims that those thresholds are biologically optimal.

## Validation commands

Run the burst fixture file:

```bash
pytest -q tests/test_burst_correctness.py
ruff check .
```

Run the signal and EDF regression cases:

```bash
pytest -q tests/test_processing.py::TestDetectSpikes::test_filtered_detection_and_waveform_edge_regression
pytest -q tests/test_parsers.py::TestParseEdfContent::test_selected_channel_calibration_and_timestamps
ruff check .
```

Run the complete repository validation:

```bash
git diff --check
ruff check .
pytest
python -m compileall .
```

## Regression validation is not biological validation

These tests answer whether the current implementation continues to produce
known outputs for controlled inputs. They protect algorithm boundaries,
calibration arithmetic, timestamp construction, refractory behavior, and
waveform-window handling from unintended code changes.

> Tests verify implementation behaviour against deterministic synthetic cases. They do not establish biological validity for every recording type, preparation, or parameter choice.

The validation suite explicitly does not establish:

- biological spike- or burst-detection sensitivity and specificity;
- suitability of default or selected parameters for an individual preparation;
- handling of every artifact found in real recordings;
- compatibility with every vendor-specific EDF variant;
- network-level or electrode-array inference from single-channel analysis;
- clinical or diagnostic validity.
