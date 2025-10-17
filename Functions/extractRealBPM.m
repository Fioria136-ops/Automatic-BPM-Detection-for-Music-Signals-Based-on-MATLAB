function [realBPM1,realBPM2,realBPM3,realBPM4,realBPM5] = extractRealBPM(fileName)
%Extract real BPM from file name
pattern = '(\d+)bpm'; %Regular expression pattern
tokens = regexp(fileName, pattern, 'tokens');
if ~isempty(tokens)
   realBPM = str2double(tokens{1}{1}); %Convert to numerical value
else
   realBPM = NaN; %Return NaN when not found
end

realBPM1 = realBPM/2;
realBPM2 = realBPM/1.5;
realBPM3 = realBPM;
realBPM4 = realBPM*1.5;
realBPM5 = realBPM*2;

end