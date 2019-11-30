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

clear;
close all;

% Parameters
wavPath1 = sprintf('./input/drums.wav'); % file path of wav signal
wavPath2 = sprintf('./input/guitar.wav'); % file path of wav signal
wavPath3 = sprintf('./input/piano.wav'); % file path of wav signal
outputDir = sprintf('./output');
A = [0.3, 0.6, -0.8; ...
    -0.2, 0.5, 0.9; ...
    -0.3, 0.6, -0.7]; % mixing matrix (3 x 3)
stepSize = 0.2; % step size parameter
maxIt = 100; % maximum number of iterations in natural gradient algorithm
type = 'laplace'; % type of score function (laplace: super-Gaussian, sech: super-Gaussian, cosh: sub-Gaussian)
backProjection = 1; % channel of back projection
drawCost = true; % draw convergence behavior of cost function values or not

% Audio read and mixing
[s1,fs] = audioread(wavPath1); % fs: sampling frequency [Hz], s1 is a vector of size "length x channels"
[s2,fs] = audioread(wavPath2); % s1, s2, and s3 are column vectors because sample wave files are monaural
[s3,fs] = audioread(wavPath3);
s = [s1.'; s2.'; s3.']; % source signal of size "3 x length"
x = A * s; % observed (mixture) signal of size "3 x length"

% ICA
[y, W, cost] = naturalGradIca(x, stepSize, maxIt, type, backProjection, drawCost);

% Output separated signals
if ~isdir(outputDir)
    mkdir(outputDir);
end
audiowrite(sprintf('%s/observedMixture.wav', outputDir), x.', fs); % observed signal
audiowrite(sprintf('%s/estimatedSignal1.wav', outputDir), y(1,:).', fs); % estimated signal 1
audiowrite(sprintf('%s/estimatedSignal2.wav', outputDir), y(2,:).', fs); % estimated signal 2
audiowrite(sprintf('%s/estimatedSignal3.wav', outputDir), y(3,:).', fs); % estimated signal 3

fprintf('The files are saved in "./output".\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%