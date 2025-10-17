clc;
clear all;

%% ////////////Initialization of paths, files, functions, etc///////////
scriptPath = fileparts(mfilename('fullpath'));%The path where the current script is located
audioFolderPath = fullfile(scriptPath, 'Dataset', 'WaivOps EDM-TR9 Open Audio Resources for Machine Learning in Music'); 
resultFilePath = fullfile(scriptPath, 'results', 'Sound_Analysis_Results.txt'); 
functionPath = fullfile(scriptPath, 'Functions');
addpath(functionPath);

audioFiles = dir(fullfile(audioFolderPath, '*.wav'));

if isempty(audioFiles)
    disp('No audio files found.');
    return;
end

resultFile = fopen(resultFilePath, 'w');
fprintf(resultFile, "Audio Analysis Results\n");
fprintf(resultFile, "=============================\n");

%% //////////////////Parameter settings/////////////////////////

sampleDuration            = 8;
frequencyRange_SF         = [500,7000];
frequencyRange_ZCR        = [2500,18000];
threshold_SF              = 1.0;
threshold_ZCR             = 1.0;
Std_factor                = 1.5;
compressionStatus         = true;
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%Warning: Opening the display figure function for large-scale detection 
%will result in generating a large number of windows!
%When displaying the verification results figure, 
%the separate display of each sample will be turned off
ploteachsample            = false;
plotresult                = true;
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

compressionThreshold      = 0.1;
compressionRatio          = 2.1;
tolerance                 = 15;  %Verify threshold
Accuracy                  = 3;   %Verify medhod
%Accuracy1:Compare only the true BPM values annotated
%in the dataset.
%Accuracy2:Compare values of the true BPM annotated
%in the dataset with their double and half values.
%Accuracy3:Compare twice, 1.5 times, half and 1/1.5
%times the actual BPM values

yTrue_all = []; %Array for storing 'real labels'
yPred_all = []; %Array for storing 'predicted labels'

if plotresult
    %Initialize chart
    figure;
    hold on;
    
    %Set coordinate axis range and function line
    x = 115:145;
    plot(x, x / 2, '--');
    plot(x, x / 1.5, '--');
    plot(x, x, '-', 'LineWidth', 1.5);
    plot(x, 1.5 * x, '--');
    plot(x, 2 * x, '--');
    ploteachsample            = false;
end

%% //////////////////Start batch processing audio files/////////////////////////
totalDeviationRate = 0;
validFileCount     = 0;

for i = 1:length(audioFiles)
    audioFileName = audioFiles(i).name;
    audioFilePath = fullfile(audioFolderPath, audioFileName);
    
    fprintf('Analyzing %s...\n', audioFileName);
    
    try
        %% Call the analysis function
        bpm = analyzeBPM_v12( ...
            audioFilePath, sampleDuration, threshold_SF, ...
            compressionThreshold, compressionRatio, compressionStatus, ...
            ploteachsample, Std_factor, threshold_ZCR,...
            frequencyRange_SF, frequencyRange_ZCR);

        %% Extract real BPM from file name
        
        [realBPM1,realBPM2,realBPM3,realBPM4,realBPM5] = extractRealBPM(audioFileName);
        if isnan(realBPM1)
            fprintf('Warning: Could not extract real BPM for %s.\n', audioFileName);
            fprintf(resultFile, 'Warning: Could not extract real BPM for %s.\n\n', audioFileName);
            continue;
        end

        %% Find the BPM that are closest to the predicted BPM
        if Accuracy == 3
            realBPMs = [realBPM1,realBPM2,realBPM3,realBPM4,realBPM5];
            [~, bestIdx]  = min(abs(bpm - realBPMs));
        closestRealBPM    = realBPMs(bestIdx);
        elseif Accuracy == 2
            realBPMs = [realBPM1,realBPM3,realBPM5];
            [~, bestIdx]  = min(abs(bpm - realBPMs));
        closestRealBPM    = realBPMs(bestIdx);
        elseif Accuracy == 1
            closestRealBPM = realBPM3;
        end


        %% Calculate errors and calculate deviation rates
        
        deviationRate = abs(bpm - closestRealBPM) / closestRealBPM * 100;
        totalDeviationRate = totalDeviationRate + deviationRate;
        validFileCount = validFileCount + 1;

        %% Determine whether it is within the tolerance range (binary classification)
        
        if abs(bpm - closestRealBPM) <= tolerance
            yPred = 1;
        else
            yPred = 0;
        end
        yTrue = 1; 

        %Collect into array
        yTrue_all(end+1) = yTrue;
        yPred_all(end+1) = yPred;

        if plotresult
            deviation = abs(bpm - closestRealBPM);
            if deviation > tolerance
                %Points beyond the tolerance range are marked in red
                plot(realBPM3, bpm, 'o', 'MarkerEdgeColor', 'r', 'MarkerSize', 5, 'LineWidth', 1.5);
            else
                %Points within the tolerance range are marked in black
                plot(realBPM3, bpm, 'o', 'MarkerEdgeColor', 'k', 'MarkerSize', 5, 'LineWidth', 1.5);
            end
        end
        %% Display&Record Results
       
        fprintf('File: %s\n', audioFileName);
        fprintf('Possible BPM: %.2f\n', bpm);
        fprintf('Real BPM: %.2f\n', realBPM3);
        fprintf('Closest real BPMs: %.2f (Deviation: %.2f%%)\n\n', closestRealBPM, deviationRate);

        fprintf(resultFile, 'File: %s\n', audioFileName);
        fprintf(resultFile, 'Possible BPM1: %.2f\n', bpm);
        fprintf(resultFile, 'Real BPM: %.2f\n', realBPM3);
        fprintf(resultFile, 'Closest real BPMs: %.2f (Deviation: %.2f%%)\n\n', closestRealBPM, deviationRate);

    catch ME
        %Capture errors and record them
        fprintf('Error analyzing %s: %s\n', audioFileName, ME.message);
        fprintf(resultFile, 'Error analyzing %s: %s\n\n', audioFileName, ME.message);
    end
end

%% ////////////////////Calculate average error&close file////////////////////// 
if validFileCount > 0
    averageDeviationRate = totalDeviationRate / validFileCount;
else
    averageDeviationRate = NaN;
end

fprintf('Average Deviation Rate: %.2f%%\n', averageDeviationRate);
fprintf(resultFile, 'Average Deviation Rate: %.2f%%\n', averageDeviationRate);

fclose(resultFile);

%% ///////////////Generate figures to view the validation results/////////////////
if plotresult
    title('BPM Prediction vs Real BPM');
    xlabel('Real BPM');
    ylabel('Predicted BPM');
    grid on;
    hold off;
end

%Binary confusion matrix
figure;
confMat = confusionmat(yTrue_all, yPred_all);  
confusionchart(confMat, ["False(0)","True(1)"]);
disp("Verification result(FN,TP) = ");
disp(confMat);
