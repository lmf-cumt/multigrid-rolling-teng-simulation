# TENG short-circuit current waveform simulation notes

## COMSOL electrostatic baseline

COMSOL is used as the fixed-charge electrostatic field baseline. The existing position sweep places the same PTFE equivalent surface charge density on the rolling balls and integrates induced charge on the A electrode group. This baseline is useful for field and induced-charge distribution, but it is not treated as direct proof that the four-electrode structure has 650 nC transferred charge.

| Electrode count | COMSOL scaled transfer charge (nC) | COMSOL peak current at 1 GOhm (uA) |
|---:|---:|---:|
| 2 | 380.000 | 3.693 |
| 4 | 386.986 | 9.941 |
| 6 | 387.711 | 15.969 |

## MATLAB calibrated waveform model

The MATLAB model uses a full reciprocating motion cycle with 100 mm stroke, 5.0 m/s^2 acceleration/deceleration, and 30 ms dwell at each end.
The N=2 main short-circuit pulse is calibrated to 8.0 uA and 380 nC. N=4 uses the measured 650 nC transfer charge, giving a second-pulse weight of 0.7105. N=6 extends the same decay trend.

| Electrode count | Events per half-cycle | Pulse weights | Modeled transfer charge (nC) | Peak short-circuit current (uA) |
|---:|---:|---|---:|---:|
| 2 | 1 | 1.0000 | 380.000 | 8.000 |
| 4 | 2 | 1.0000, 0.7105 | 650.000 | 8.000 |
| 6 | 3 | 1.0000, 0.7105, 0.5048 | 841.842 | 8.009 |

## Interpretation

The PTFE surface charge density is kept fixed for N=2, N=4, and N=6. The multi-grid alternating electrode pattern increases the number of effective charge-transfer events within the same mechanical cycle, so total transferred charge rises while the largest short-circuit current peak remains approximately unchanged.

Generated files are in `results_current_waveforms_matlab/`.
