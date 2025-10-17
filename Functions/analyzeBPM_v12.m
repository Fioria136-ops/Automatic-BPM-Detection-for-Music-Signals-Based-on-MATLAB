function bpm = analyzeBPM_v12(audioFilePath, sampleDuration, threshold_SF, ...
    compressionThreshold, compressionRatio, compressionStatus, ...
    ploteachsample, Std_factor, threshold_ZCR, frequencyRange_SF, frequencyRange_ZCR)
% This is the core BPM analysis script that calculates the three possible 
% BPM values by separately analyzing SF and ZCR methods and applying
% standard deviation filtering.

%% ///////// Read and Preprocess Audio Files /////////
[x, fs] = audioread(audioFilePath);
x = x(:, 1); 

%Normalize
x_current = sqrt(mean(x.^2));
targetRMS = 0.3;
x_gain = targetRMS / x_current;
x = x * x_gain;

%Extract the interval with the highest average energy
window_size = round(sampleDuration * fs);
step_size = round(4 * fs);
windows = floor((length(x) - window_size) / step_size) + 1;

if window_size < length(x) % Only the interval with the highest average energy
                           % when the sampleDuration shorter than the
                           % x

window_rms = zeros(1, windows);
for i = 1:windows
    window_start = (i - 1) * step_size + 1;
    window_end = min(window_start + window_size - 1, length(x));
    window = x(window_start:window_end);
    window_rms(i) = sqrt(mean(window.^2)); % RMS calculation
end

[~, max_energy_window_idx] = max(window_rms);
max_window_start = (max_energy_window_idx - 1) * step_size + 1;

% Crop audio to maximum energy window
sample_start = max_window_start;
sample_end = min(length(x), sample_start + window_size - 1);
x = x(sample_start:sample_end);

else
end

% Apply Dynamic Range Compression
if compressionStatus
   x = dynamicRangeCompression(x, compressionThreshold, compressionRatio);
   disp('Compressor is available');
else
   disp('Compressor is not available');
end

%% ///////// Onset Detection Using SF and ZCR /////////
[onsetTime_SF, ~, sf, t_SF, Threshold_SF] = Onsetsdetect_SF(x, fs, threshold_SF, frequencyRange_SF);
[onsetTime_ZCR, ~, zcr, t_ZCR, Threshold_ZCR] = Onsetsdetect_ZCR(x, fs, threshold_ZCR, frequencyRange_ZCR);

%% ///////// Calculate BPM for SF /////////
if length(onsetTime_SF) > 1
    timeDiffs_SF = diff(onsetTime_SF);

    %std filter
    mean_diff_SF = mean(timeDiffs_SF);
    std_diff_SF = std(timeDiffs_SF);
    filtered_diffs_SF = timeDiffs_SF(abs(timeDiffs_SF - mean_diff_SF) <= Std_factor * std_diff_SF);

    if ~isempty(filtered_diffs_SF)
        avg_time_diff_SF = mean(filtered_diffs_SF); % average Time diff(filtered)
        bpm_SF = 60 / avg_time_diff_SF;
    else
        bpm_SF = NaN;
    end
else
    bpm_SF = NaN;
    filtered_diffs_SF = [];
end

%% ///////// Calculate BPM for ZCR /////////
if length(onsetTime_ZCR) > 1
   timeDiffs_ZCR = diff(onsetTime_ZCR);

   %std filter
   mean_diff_ZCR = mean(timeDiffs_ZCR);
   std_diff_ZCR = std(timeDiffs_ZCR);
   filtered_diffs_ZCR = timeDiffs_ZCR(abs(timeDiffs_ZCR - mean_diff_ZCR) <= Std_factor * std_diff_ZCR);

    if ~isempty(filtered_diffs_ZCR)%if is not empty
        avg_time_diff_ZCR = mean(filtered_diffs_ZCR);
        bpm_ZCR = 60 / avg_time_diff_ZCR;
    else%if is empty
        bpm_ZCR = NaN;
    end
else
    bpm_ZCR = NaN;
    filtered_diffs_ZCR = [];
end

%% ///////// Calculate Uniformity /////////
if ~isempty(filtered_diffs_SF)
    std_diff_SF = std(filtered_diffs_SF); % Standard deviation of SF
    if std_diff_SF > 0
        uniformity_SF = 1 / std_diff_SF; % Uniformity for SF
    else
        uniformity_SF = 0;
    end
else
    uniformity_SF = 0; % If filtered_diffs_SF is empty
end

if ~isempty(filtered_diffs_ZCR)
    std_diff_ZCR = std(filtered_diffs_ZCR); % Standard deviation of ZCR
    if std_diff_ZCR > 0
        uniformity_ZCR = 1 / std_diff_ZCR; % Uniformity for ZCR
    else
        uniformity_ZCR = 0; 
    end
else
    uniformity_ZCR = 0; % If filtered_diffs_ZCR is empty
end

%% ///////// Assign Weights Based on Uniformity /////////
if std_diff_SF == 0 && std_diff_ZCR == 0
    % Both standard deviations are zero
    w_SF = 0.5;
    w_ZCR = 0.5;
elseif std_diff_SF == 0
    % SF's standard deviation is zero
    w_SF = 1;
    w_ZCR = 0;
elseif std_diff_ZCR == 0
    % ZCR's standard deviation is zero
    w_SF = 0;
    w_ZCR = 1;
else
    % Normal case: both std_diff_SF and std_diff_ZCR are greater than zero
    total_uniformity = uniformity_SF + uniformity_ZCR;
    w_SF = uniformity_SF / total_uniformity;
    w_ZCR = uniformity_ZCR / total_uniformity;
end

disp(['Adjusted SF weight: ', num2str(w_SF)]);
disp(['Adjusted ZCR weight: ', num2str(w_ZCR)]);

%% ////////////////////Calculate the weighted BPM////////////////////////////
%This section is used to improve robustness by multiplying some detection results 
%below the typical value by a certain value to place them in the typical value range.
if bpm_SF < 20
    bpm_SF = bpm_SF*8;
elseif bpm_SF < 40
    bpm_SF = bpm_SF*4;
elseif bpm_SF < 80
    bpm_SF = bpm_SF*2;
else
end

if bpm_ZCR < 20
    bpm_ZCR = bpm_ZCR*8;
elseif bpm_ZCR < 40
    bpm_ZCR = bpm_ZCR*4;
elseif bpm_ZCR < 80
    bpm_ZCR = bpm_ZCR*2;
else
end
% Calculate the weighted BPM
if isnan(bpm_SF)
    bpm = bpm_ZCR;
elseif isnan(bpm_ZCR)
    bpm = bpm_SF;
else
    bpm = w_SF * bpm_SF + w_ZCR * bpm_ZCR;
end

%% ///////// Plot Results (Optional) /////////
if ploteachsample
    figure;
    subplot(3, 1, 1);
    plot(t_SF, sf, 'b');
    hold on;
    scatter(onsetTime_SF, sf(round(onsetTime_SF * length(sf) / t_SF(end))), 'r*');
    hold on;
    yline(Threshold_SF, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Threshold');
    title('SF Method - Spectral Flux');
    xlabel('Time (s)');
    ylabel('Spectral Flux');
    hold off;

    subplot(3, 1, 2);
    plot(t_ZCR, zcr, 'b');
    hold on;
    scatter(onsetTime_ZCR, zcr(round(onsetTime_ZCR * length(zcr) / t_ZCR(end))), 'r*');
    hold on
    yline(Threshold_ZCR, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Threshold');
    title('ZCR Method - Zero Crossing Rate');
    xlabel('Time (s)');
    ylabel('Zero Crossing Rate');
    hold off;

    subplot(3, 1, 3);
    plot(x, 'k');
    hold on;
    scatter(onsetTime_SF * fs, zeros(size(onsetTime_SF))+1, 'g*', 'LineWidth', 1.5);
    scatter(onsetTime_ZCR * fs, zeros(size(onsetTime_ZCR))-1, 'm*', 'LineWidth', 1.5);
    title('SF and ZCR Onsets');
    xlabel('Sample Index');
    ylabel('Amplitude');
    hold off;
end
end
