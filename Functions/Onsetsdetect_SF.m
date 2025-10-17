function [onsetTime, onsetValues, sf, t, Threshold] = Onsetsdetect_SF(x, fs, threshold, frequencyRange)
% This function finds the onset points based on Spectral Flux (SF)
% Modified to use SF(n) = sum_k H(|X(n,k)| - |X(n-1,k)|)

% Calculate the spectrogram
windowSize = 2048; % STFT window size
hopSize = 1024;    % STFT hop size

% BandPass
[b, a] = butter(2, frequencyRange / (fs / 2), 'bandpass');
x = filter(b, a, x);

overlap = windowSize - hopSize; % Overlap size
[X, ~, t] = stft(x, fs, 'Window', hamming(windowSize, 'periodic'), 'OverlapLength', overlap, 'FFTLength', windowSize);
X = abs(X); % Magnitude of the spectrogram

% Calculate SF using the specified formula
sf = zeros(size(X, 2), 1); 
for n = 2:size(X, 2)
    sf(n) = sum(max(X(:, n) - X(:, n-1), 0)); 
end

% Normalize SF to 0~1
sf = rescale(sf);

% Dynamic threshold
Threshold = threshold * mean(sf);

%Fws
std_sf = std(sf);
fws = 100 * std_sf;

% Detect local maxima
lmax = islocalmax(sf, "MinSeparation", fws);

% Extract indices of local maxima
idx = find(lmax);

% Extract times and values of local maxima
lMaxValues = sf(idx);
lMaxTimes = t(idx);

% Filter onsets using the dynamic threshold
validOnsets = lMaxValues > Threshold;

% Remaining onsets
onsetTime = lMaxTimes(validOnsets);
onsetValues = lMaxValues(validOnsets);
end
