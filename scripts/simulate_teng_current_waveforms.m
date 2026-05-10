function simulate_teng_current_waveforms()
%SIMULATE_TENG_CURRENT_WAVEFORMS
% Generate full-cycle short-circuit current waveforms for N=2, N=4, and
% N=6 rolling-ball TENG structures. COMSOL field results are used as the
% electrostatic baseline, while this MATLAB postprocess applies the
% experimentally calibrated multi-transfer current model.

rootDir = fileparts(fileparts(mfilename('fullpath')));
outDir = fullfile(rootDir, 'results_current_waveforms_matlab');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

p.electrodeWidth = 16e-3;
p.electrodeGap = 5e-3;
p.pcbThickness = 1e-3;
p.ballDiameter = 4e-3;
p.ballCount = 4;
p.parallelUnits = 16 * 9;
p.surfaceChargeDensity = -50e-6;
p.stroke = 100e-3;
p.acceleration = 5.0;
p.dwellStart = 30e-3;
p.dwellStop = 30e-3;
p.samples = 12000;
p.qN2 = 380e-9;
p.qN4 = 650e-9;
p.peakShortCircuitCurrent = 8e-6;

[time, position, velocity] = buildMotion(p);
electrodeCounts = [2, 4, 6];
colors = [0.184, 0.435, 0.624; 0.204, 0.659, 0.325; 0.698, 0.416, 0.000];
results = struct([]);

for idx = 1:numel(electrodeCounts)
    n = electrodeCounts(idx);
    eventCount = max(1, n / 2);
    weights = pulseWeights(eventCount, p);
    targetCharge = modeledTransferCharge(n, p);
    [current, charge, centersForward, centersReverse, pulseTau] = ...
        shortCircuitWaveform(time, eventCount, weights, targetCharge, p);

    result.n = n;
    result.eventCount = eventCount;
    result.weights = weights;
    result.time = time;
    result.position = position;
    result.velocity = velocity;
    result.current = current;
    result.charge = charge;
    result.centersForward = centersForward;
    result.centersReverse = centersReverse;
    result.pulseTau = pulseTau;
    result.modeledCharge = targetCharge;
    results = [results; result]; %#ok<AGROW>

    tableOut = table(time(:), position(:), velocity(:), charge(:), current(:), ...
        'VariableNames', {'time_s', 'position_m', 'motion_velocity_m_per_s', ...
        'source_charge_C', 'short_circuit_current_A'});
    writetable(tableOut, fullfile(outDir, sprintf('short_circuit_current_N%d.csv', n)));
    plotSingleCurrent(fullfile(outDir, sprintf('short_circuit_current_N%d_full_cycle.svg', n)), result, colors(idx, :));
end

writeSummary(fullfile(outDir, 'short_circuit_current_summary.csv'), results);
plotComparison(fullfile(outDir, 'short_circuit_current_comparison_full_cycle.svg'), results, colors);
writeNotes(fullfile(outDir, 'current_waveform_simulation_notes.md'), rootDir, p, results);

fprintf('Generated short-circuit waveform results in %s\n', outDir);
end

function [time, position, velocity] = buildMotion(p)
tAcc = sqrt(p.stroke / p.acceleration);
tMotion = 2.0 * tAcc;
duration = p.dwellStart + tMotion + p.dwellStop + tMotion + p.dwellStart;
time = linspace(0.0, duration, p.samples).';
position = zeros(size(time));
velocity = zeros(size(time));

t0 = p.dwellStart;
t1 = t0 + tAcc;
t2 = t0 + tMotion;
t3 = t2 + p.dwellStop;
t4 = t3 + tAcc;
t5 = t3 + tMotion;
vPeak = p.acceleration * tAcc;

mask = time >= t0 & time < t1;
tau = time(mask) - t0;
position(mask) = 0.5 * p.acceleration * tau.^2;
velocity(mask) = p.acceleration * tau;

mask = time >= t1 & time < t2;
tau = time(mask) - t1;
position(mask) = 0.5 * p.stroke + vPeak * tau - 0.5 * p.acceleration * tau.^2;
velocity(mask) = vPeak - p.acceleration * tau;

mask = time >= t2;
position(mask) = p.stroke;

mask = time >= t3 & time < t4;
tau = time(mask) - t3;
position(mask) = p.stroke - 0.5 * p.acceleration * tau.^2;
velocity(mask) = -p.acceleration * tau;

mask = time >= t4 & time < t5;
tau = time(mask) - t4;
position(mask) = 0.5 * p.stroke - vPeak * tau + 0.5 * p.acceleration * tau.^2;
velocity(mask) = -vPeak + p.acceleration * tau;

mask = time >= t5;
position(mask) = 0.0;
velocity(mask) = 0.0;
end

function weights = pulseWeights(eventCount, p)
if eventCount <= 1
    weights = 1.0;
else
    r = p.qN4 / p.qN2 - 1.0;
    weights = r .^ (0:(eventCount - 1));
end
end

function q = modeledTransferCharge(n, p)
weights = pulseWeights(max(1, n / 2), p);
q = p.qN2 * sum(weights);
end

function [current, charge, centersForward, centersReverse, pulseTau] = shortCircuitWaveform(time, eventCount, weights, targetCharge, p)
tAcc = sqrt(p.stroke / p.acceleration);
tMotion = 2.0 * tAcc;
forwardStart = p.dwellStart;
forwardEnd = forwardStart + tMotion;
reverseStart = forwardEnd + p.dwellStop;
reverseEnd = reverseStart + tMotion;
centersForward = forwardStart + ((0:(eventCount - 1)) + 0.5) ./ eventCount .* tMotion;
centersReverse = reverseStart + ((0:(eventCount - 1)) + 0.5) ./ eventCount .* tMotion;

pulseTau = fitTrainPulseWidth(time, centersForward, centersReverse, weights, targetCharge, ...
    p.peakShortCircuitCurrent, forwardStart, forwardEnd, reverseStart, reverseEnd);
forward = weightedPulses(time, centersForward, weights, pulseTau);
reverse = weightedPulses(time, centersReverse, weights, pulseTau);
current = p.peakShortCircuitCurrent .* (forward - reverse);

dwellMask = time < forwardStart | (time > forwardEnd & time < reverseStart) | time > reverseEnd;
current(dwellMask) = current(dwellMask) .* 0.02;
charge = cumtrapz(time, current);
end

function pulseTau = fitTrainPulseWidth(time, centersForward, centersReverse, weights, targetCharge, targetPeak, forwardStart, forwardEnd, reverseStart, reverseEnd)
lo = 1e-4;
hi = max(0.5, time(end) - time(1));
for k = 1:80 %#ok<NASGU>
    mid = sqrt(lo * hi);
    forward = weightedPulses(time, centersForward, weights, mid);
    reverse = weightedPulses(time, centersReverse, weights, mid);
    trial = targetPeak .* (forward - reverse);
    dwellMask = time < forwardStart | (time > forwardEnd & time < reverseStart) | time > reverseEnd;
    trial(dwellMask) = trial(dwellMask) .* 0.02;
    area = trapz(time, max(trial, 0));
    if area < targetCharge
        lo = mid;
    else
        hi = mid;
    end
end
pulseTau = hi;
end

function y = rawPulses(time, centers, pulseTau)
z = (time(:) - centers(:).') ./ pulseTau;
z = max(min(z, 30.0), -30.0);
y = sum(1.0 ./ cosh(z).^2, 2);
end

function y = weightedPulses(time, centers, weights, pulseTau)
z = (time(:) - centers(:).') ./ pulseTau;
z = max(min(z, 30.0), -30.0);
y = sum(weights(:).' ./ cosh(z).^2, 2);
end

function writeSummary(path, results)
electrode_count = zeros(numel(results), 1);
event_count = zeros(numel(results), 1);
modeled_total_transfer_charge_nC = zeros(numel(results), 1);
integrated_positive_charge_nC = zeros(numel(results), 1);
integrated_negative_charge_nC = zeros(numel(results), 1);
current_peak_positive_uA = zeros(numel(results), 1);
current_peak_negative_uA = zeros(numel(results), 1);
current_peak_abs_uA = zeros(numel(results), 1);
pulse_tau_s = zeros(numel(results), 1);
pulse_weights = strings(numel(results), 1);
duration_s = zeros(numel(results), 1);

for i = 1:numel(results)
    r = results(i);
    electrode_count(i) = r.n;
    event_count(i) = r.eventCount;
    modeled_total_transfer_charge_nC(i) = r.modeledCharge * 1e9;
    integrated_positive_charge_nC(i) = trapz(r.time, max(r.current, 0)) * 1e9;
    integrated_negative_charge_nC(i) = trapz(r.time, min(r.current, 0)) * 1e9;
    current_peak_positive_uA(i) = max(r.current) * 1e6;
    current_peak_negative_uA(i) = min(r.current) * 1e6;
    current_peak_abs_uA(i) = max(abs(r.current)) * 1e6;
    pulse_tau_s(i) = r.pulseTau;
    pulse_weights(i) = join(string(compose('%.4f', r.weights)), ',');
    duration_s(i) = r.time(end) - r.time(1);
end

summary = table(electrode_count, event_count, modeled_total_transfer_charge_nC, ...
    integrated_positive_charge_nC, integrated_negative_charge_nC, ...
    current_peak_positive_uA, current_peak_negative_uA, current_peak_abs_uA, ...
    pulse_tau_s, pulse_weights, duration_s);
writetable(summary, path);
end

function plotComparison(path, results, colors)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 900, 470]);
hold on;
for i = 1:numel(results)
    plot(results(i).time, results(i).current * 1e6, 'LineWidth', 1.6, 'Color', colors(i, :));
end
grid on;
xlabel('Time (s)');
ylabel('Short-circuit current (\muA)');
title('Short-circuit current comparison, full reciprocating cycle');
legend({'N=2', 'N=4', 'N=6'}, 'Location', 'northeast');
exportgraphics(fig, path, 'ContentType', 'vector');
close(fig);
end

function plotSingleCurrent(path, result, color)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 900, 470]);
plot(result.time, result.current * 1e6, 'LineWidth', 1.6, 'Color', color);
grid on;
xlabel('Time (s)');
ylabel('Short-circuit current (\muA)');
title(sprintf('N=%d short-circuit current, full reciprocating cycle', result.n));
exportgraphics(fig, path, 'ContentType', 'vector');
close(fig);
end

function writeNotes(path, rootDir, p, results)
comsolSummaryPath = fullfile(rootDir, 'results_comsol_field', 'comsol_field_summary_1G.csv');
comsolLines = {};
if exist(comsolSummaryPath, 'file')
    t = readtable(comsolSummaryPath);
    comsolLines{end + 1} = '| Electrode count | COMSOL scaled transfer charge (nC) | COMSOL peak current at 1 GOhm (uA) |'; %#ok<AGROW>
    comsolLines{end + 1} = '|---:|---:|---:|'; %#ok<AGROW>
    for i = 1:height(t)
        comsolLines{end + 1} = sprintf('| %d | %.3f | %.3f |', ...
            t.electrode_count(i), t.comsol_scaled_transfer_charge_nC(i), t.current_peak_uA_1G(i)); %#ok<AGROW>
    end
else
    comsolLines{end + 1} = 'COMSOL baseline file was not found; MATLAB waveform generation used the calibrated transfer model only.'; %#ok<AGROW>
end

summaryLines = {};
summaryLines{end + 1} = '| Electrode count | Events per half-cycle | Pulse weights | Modeled transfer charge (nC) | Peak short-circuit current (uA) |'; %#ok<AGROW>
summaryLines{end + 1} = '|---:|---:|---|---:|---:|'; %#ok<AGROW>
for i = 1:numel(results)
    r = results(i);
    summaryLines{end + 1} = sprintf('| %d | %d | %s | %.3f | %.3f |', ...
        r.n, r.eventCount, strjoin(cellstr(compose('%.4f', r.weights)), ', '), ...
        r.modeledCharge * 1e9, max(abs(r.current)) * 1e6); %#ok<AGROW>
end

lines = [
    "# TENG short-circuit current waveform simulation notes"
    ""
    "## COMSOL electrostatic baseline"
    ""
    "COMSOL is used as the fixed-charge electrostatic field baseline. The existing position sweep places the same PTFE equivalent surface charge density on the rolling balls and integrates induced charge on the A electrode group. This baseline is useful for field and induced-charge distribution, but it is not treated as direct proof that the four-electrode structure has 650 nC transferred charge."
    ""
    string(comsolLines(:))
    ""
    "## MATLAB calibrated waveform model"
    ""
    sprintf("The MATLAB model uses a full reciprocating motion cycle with %.0f mm stroke, %.1f m/s^2 acceleration/deceleration, and %.0f ms dwell at each end.", p.stroke * 1e3, p.acceleration, p.dwellStart * 1e3)
    sprintf("The N=2 main short-circuit pulse is calibrated to %.1f uA and %.0f nC. N=4 uses the measured %.0f nC transfer charge, giving a second-pulse weight of %.4f. N=6 extends the same decay trend.", p.peakShortCircuitCurrent * 1e6, p.qN2 * 1e9, p.qN4 * 1e9, p.qN4 / p.qN2 - 1.0)
    ""
    string(summaryLines(:))
    ""
    "## Interpretation"
    ""
    "The PTFE surface charge density is kept fixed for N=2, N=4, and N=6. The multi-grid alternating electrode pattern increases the number of effective charge-transfer events within the same mechanical cycle, so total transferred charge rises while the largest short-circuit current peak remains approximately unchanged."
    ""
    "Generated files are in `results_current_waveforms_matlab/`."
    ];
fid = fopen(path, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', lines);
clear cleanup;
end
