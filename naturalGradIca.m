function [y,W,cost] = naturalGradIca(x,stepSize,maxIt,type,backProjection,drawCost,initW)
%
% naturalGradIca: Blind source separation using ICA based on natural Grad.
%
% Coded by D. Kitamura (d-kitamura@ieee.org)
%
% # Original paper
% S. Amari, "Natural gradient works efficiently in learning," Neural
% Computation, vol. 10, no. 2, pp. 251-276, 1998.
%
% see also
% http://d-kitamura.net
%
% [syntax]
%   [y,W,cost] = naturalGradIca(x)
%   [y,W,cost] = naturalGradIca(x,stepSize)
%   [y,W,cost] = naturalGradIca(x,stepSize,maxIt)
%   [y,W,cost] = naturalGradIca(x,stepSize,maxIt,type)
%   [y,W,cost] = naturalGradIca(x,stepSize,maxIt,type,backProjection)
%   [y,W,cost] = naturalGradIca(x,stepSize,maxIt,type,backProjection,drawCost)
%   [y,W,cost] = naturalGradIca(x,stepSize,maxIt,type,backProjection,drawCost,initW)
%
% [inputs]
%              x: mixture signal (sources x time samples)
%       stepSize: initial step size for natural gradient algorithm (default: 1)
%          maxIt: maximum number of iterations (default: 100)
%           type: choose score function from below (default: laplace)
%                 'laplace' : super-Gaussian, y/abs(y)
%                 'sech'    : super-Gaussian, tanh(y)
%                 'cosh'    : sub-Gaussian, y-tanh(y)
% backProjection: channel of applying back projection (0: do not apply, number: projection channel, default: 1)
%       drawCost: show convergence behavior of cost function values or not (true or false, default: false)
%          initW: initial demixing matrix (sources x channels, square matrix)
%
% [outputs]
%              y: estimated signals (sources x time samples)
%              W: estimated demixing matrix (sources x channels)
%           cost: convergence behavior of cost function values (maxIt+1 x 1)
%

% Check errors and set default values
[nCh,nSample] = size(x);
if (nCh == 1)
    error('The input mixture signal must be monaural.\n');
elseif (nCh > nSample)
    x = x.'; % transpose x because definition of x is wrong
end
if (nargin < 2)
    stepSize = 1; % default value of step size
end
if (nargin < 3)
    maxIt = 100; % default value of maximum number of iterations
end
if (nargin < 4)
    type = 'laplace'; % default score function
elseif ~strcmp(type, 'laplace') && ~strcmp(type, 'sech') && ~strcmp(type, 'cosh')
    error('Input type (score function) is not supported.\n');
end
if (nargin < 5)
    backProjection = 1; % default backProjection (apply back projection onto 1st channel)
end
if (nargin < 6)
    drawCost = true; % default drawCost (do not show convergence behavior)
end
if (nargin < 7)
    W = randn(nCh); % initialize W with random values
else
    if size(initW,1) ~= nCh || size(initW,2) ~= nCh || size(initW,3) ~= 1
        error('The size of input initial W might be wrong.\n');
    end
    W = initW; % initialize W with input values
end

% ICA iteration based on natural gradient algorithm
I = eye(nCh); % identity matrix
y = W * x; % initial estimated signal
if drawCost
    cost = zeros(maxIt+1,1); % memory allocation
    cost(1,1) = calcCost_local(W, y, nSample, type); % initial cost value
else
    cost = 0;
end
fprintf('Iteration:     ');
for it = 1:maxIt
    sy = scoreFunc_local(y, type); % score function value
    E = (sy * y.') / nSample; % sum of inner product values of sy and y, E is a matrix of size "nCh x nCh"
    W = W - stepSize * (E - I) * W; % update rule based on natural gradient
    y = W * x; % update of separated signal
    if drawCost
        cost(it+1,1) = calcCost_local(W, y, nSample, type); % calculate cost function value in ICA
    end
    fprintf('\b\b\b\b%4d', it); % display current iteration number
end

% Draw cost function convergence
if drawCost
    plot((0:maxIt),cost);
    set(gca, 'FontName', 'Arial', 'FontSize', 14);
    xlabel('Number of iteration', 'FontSize', 15);
    ylabel('Value of cost function', 'FontSize', 15);
end

% Apply back projection or normalization
if backProjection ~= 0
    if backProjection > nCh
        error('Value of backProjection is incorrect.\n');
    else
        B = (x*y')/(y*y'); % closed-form solution of min_D |x-Dy|^2
        y = (B(backProjection,:).').*y; % using implicit expansion
        W = diag(B(backProjection,:))*W; % demixing matrix after applying back projection
    end
else
    y = y ./ max(max(abs(y))); % Since output scale of ICA is undetermined, apply normalization
    W = max(max(abs(y)))*W;
end
fprintf(' Natural Gradient ICA done.\n');
end

% Local functions
function sy = scoreFunc_local(y, type) % calculate score function (-log p(y))'
if strcmp(type, 'laplace')
    sy = y./abs(y);
elseif strcmp(type, 'sech')
    sy = tanh(y);
elseif strcmp(type, 'cosh')
    sy = y-tanh(y);
end
end

function cost = calcCost_local(W, y, T, type) % calculate ICA cost function value
if strcmp(type, 'laplace')
    py = (1/2)*exp(-1*abs(y)); % likelihood
elseif strcmp(type, 'sech')
    py = sech(y)/pi; % likelihood
elseif strcmp(type, 'cosh')
    py = exp(-y.^2/2).*cosh(y); % likelihood
end
cost = -log(abs(det(W))) - (1/T)*sum(sum(log(py))); % cost function in ICA
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%