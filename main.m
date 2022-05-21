%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample program for blind source separation using independent component  %
% analysis (ICA) based on natural gradient algorithm                      %
%                                                                         %
% Coded by D. Kitamura (d-kitamura@ieee.org)                              %
%                                                                         %
% # Original paper                                                        %
% S. Amari, "Natural gradient works efficiently in learning," Neural      %
% Computation, vol. 10, no. 2, pp. 251-276, 1998.                         %
%                                                                         %
% See also:                                                               %
% http://d-kitamura.net                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc;

% Parameters
wavPath(1) = "./input/drums.wav"; % file path of wav signal
wavPath(2) = "./input/guitar.wav"; % file path of wav signal
wavPath(3) = "./input/piano.wav"; % file path of wav signal
outputDir = "./output/";
mixMat = [0.3, 0.6, -0.8; ...
         -0.2, 0.5, 0.9; ...
         -0.3, 0.6, -0.7]; % mixing matrix (3 x 3)
stepSize = 0.1; % step size parameter
maxIt = 100; % maximum number of iterations in natural gradient algorithm
type = "LAP"; % type of score function ("LAP": super-Gaussian, "SEC": super-Gaussian, "COS": sub-Gaussian)
refMic = 1; % reference channel for back projection technique
isDrawCost = true; % draw convergence behavior of cost function values or not

% Read source signals
for iSrc = 1:numel(wavPath)
    [srcSig(:, iSrc), fs] = audioread(wavPath(iSrc)); % srcSig: "signal length x channels", fs: sampling frequency [Hz], 
end

% Mix source signals
obsSig = mixMat * srcSig.'; % observed (mixture) signal of size "3 x length"
obsSig = obsSig.'; % time samples x 3

% Apply ICA
[estSig, demixMat, cost] = naturalGradIca(obsSig, "stepSize", stepSize, "nIter", maxIt, "srcType", type, "chBackProj", refMic, "isPlot", isDrawCost);

% Output estimated signals
if ~isfolder(outputDir); mkdir(outputDir); end
audiowrite(outputDir + "obsSig.wav", obsSig, fs); % observed signal
audiowrite(outputDir + "estSig1.wav", estSig(:, 1), fs); % estimated signal 1
audiowrite(outputDir + "estSig2.wav", estSig(:, 2), fs); % estimated signal 2
audiowrite(outputDir + "estSig3.wav", estSig(:, 3), fs); % estimated signal 3
fprintf("The files are saved in " + outputDir + ".\n");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%