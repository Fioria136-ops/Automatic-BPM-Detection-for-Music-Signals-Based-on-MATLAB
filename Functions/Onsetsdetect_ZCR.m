function [onsetTime, onsetValues, zcr, t, Threshold] = Onsetsdetect_ZCR(x, fs, threshold, frequencyRange)
%this function find the onset points base on ZCR
%Window Setting
windowSize = 2048; %WindowSize
hopSize = 1024;    %Step length

%BandPass
[b, a] = butter(2, frequencyRange / (fs / 2), 'bandpass');
x = filter(b, a, x);

%initialization
numFrames = floor((length(x) - windowSize) / hopSize) + 1;
zcr = zeros(1, numFrames);
t = (0:numFrames-1) * (hopSize / fs);

%Calculate ZCR
for n = 1:numFrames
    startIdx = (n - 1) * hopSize + 1;
    endIdx = startIdx + windowSize - 1;
    if endIdx > length(x)
        break;
    end
    frame = x(startIdx:endIdx);
    zcr(n) = sum(abs(diff(sign(frame)))) / (2 * length(frame));
end

%Normalize to 0~1
zcr = (zcr - min(zcr)) / (max(zcr) - min(zcr)); 

%Threshold setting
Threshold = threshold * mean(zcr); 

%Fws
std_zcr = std(zcr);
fws = 2 * std_zcr;

%Detect localmax
lmax = islocalmax(zcr, 'MinProminence', fws);

%extract
idx = find(lmax);
onsetValues = zcr(idx);
onsetTime = t(idx);
end