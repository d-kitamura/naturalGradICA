function [estSig, demixMat, cost] = naturalGradIca(obsSig, args)
% naturalGradIca: Blind source separation using ICA based on natural Grad.
% Coded by D. Kitamura (d-kitamura@ieee.org)
%
% [syntax]
%   [estSig, demixMat, cost] 
%        = naturalGradIca(obsSig, "stepSize", 0.1, "nIter", 100, 
%                         "srcType", "LAP", "chBackProj", 1, 
%                         "isPlot", false, "demixMat", randn(size(x, 2)))
%
% [inputs]
%     obsSig: observed mixture signal (time samples x channels)
%   stepSize: initial step size for natural gradient algorithm (default: 1)
%      nIter: maximum number of iterations (default: 100)
%    srcType: choose score function from below (default: "LAP")
%             "LAP" : super-Gaussian (laplace dist.), y/abs(y)
%             "SEC"    : super-Gaussian (sech dist.), tanh(y)
%             "COS"    : sub-Gaussian (cosh dist.), y-tanh(y)
% chBackProj: channel of applying back projection (0: do not apply, number: projection channel, default: 1)
%     isPlot: show convergence behavior of cost function values or not (true or false, default: false)
%   demixMat: initial demixing matrix (sources x channels, square matrix)
%
% [outputs]
%     estSig: estimated signals (time samples x sources)
%   demixMat: estimated demixing matrix (sources x channels)
%       cost: convergence behavior of cost function values (args.nIter+1 x 1)
%

% Check errors and set default values
arguments
    obsSig (:, :) double
    args.stepSize (1, 1) double {mustBePositive} = 0.1
    args.nIter (1, 1) double {mustBePositive, mustBeInteger} = 100
    args.srcType (1, 1) string {mustBeMember(args.srcType, ["LAP", "SEC", "COS"])} = "LAP"
    args.chBackProj (1, 1) double {mustBeNonnegative, mustBeInteger} = 1
    args.isPlot (1, 1) logical = false
    args.demixMat (:, :) double = randn(size(obsSig, 2))
end
nCh = size(obsSig, 2);
if (nCh == 1); error("x must be multichannel signal.\n"); end
if ~all(size(args.demixMat) == [nCh, nCh], "all"); error("The size of initial demixing matrix is wrong.\n"); end

% ICA iteration based on natural gradient algorithm
[estSig, demixMat, cost] = local_naturalGradIca(obsSig, args.demixMat, args.stepSize, args.nIter, args.srcType, args.isPlot);

% Plot cost function behavior
if args.isPlot; local_plotCost(cost); end

% Apply back projection or normalization
[estSig, demixMat] = local_backProjection(estSig, obsSig, demixMat, args.chBackProj);

% Check estimated signals
if anynan(estSig); error("The estimated signal includes NaN values. Please adjust step size parameter.\n"); end
end

%% Local functions
%--------------------------------------------------------------------------
function [y, W, cost] = local_naturalGradIca(x, W, mu, nIter, type, isPlot)
% Initialization
x = x.';
[nCh, nSample] = size(x);
I = eye(nCh); % identity matrix
y = W*x; % initial estimated signal
cost = zeros(nIter+1,1); % memory allocation
if isPlot; cost(1,1) = local_calcCost(W, y, nSample, type); end

% Normalization
normCoef = max(abs(x), [], "all");
x = x/normCoef;

% Update iteration based on natural gradient algorithm
fprintf("Iteration:     ");
for iIter = 1:nIter
    sy = local_scoreFunc(y, type); % score function value
    E = (sy*y.') / nSample; % sum of inner product values of sy and y, E is a matrix of size "nCh x nCh"
    W = W - mu*(E-I)*W; % update rule based on natural gradient
    y = W*x; % update estimated signal

    if isPlot; cost(iIter+1, 1) = local_calcCost(W, y, nSample, type); end
    fprintf("\b\b\b\b%4d", iIter); % display current iteration number
end
y = y.'; % sources x time samples

% Denormalization
y = y*normCoef;

fprintf(" Natural Gradient ICA done.\n");
end

%--------------------------------------------------------------------------
function sy = local_scoreFunc(y, type) % calculate score function (-log p(y))'
if type == "LAP" % Laplace
    sy = y./abs(y);
elseif type == "SEC" % sech
    sy = tanh(y);
elseif type == "COS" % cosh
    sy = y-tanh(y);
end
end

%--------------------------------------------------------------------------
function cost = local_calcCost(W, y, T, type) % calculate ICA cost function value
if type == "LAP" % Laplace
    py = (1/2)*exp(-1*abs(y));
elseif type == "SEC" % sech
    py = sech(y)/pi;
elseif type == "COS" % cosh
    py = exp(-y.^2/2).*cosh(y);
end
cost = -log(abs(det(W))) - (1/T)*sum(log(py), "all"); % cost function in ICA (negative log likelihood)
end

%--------------------------------------------------------------------------
function [y, W] = local_backProjection(y, x, W, ch)
y = y.'; % sources x time samples
x = x.'; % channels x time samples
nCh = size(x, 1);
if ch ~= 0 % apply back projection technique
    if ch > nCh
        error("Channel for back projection technique is incorrect.\n");
    else
        D = (x*y')/(y*y'); % closed-form solution of min_D |x-Dy|^2
        y = (D(ch, :).').*y; % using implicit expansion
        W = diag(D(ch, :))*W; % demixing matrix after applying back projection
    end
else % Since output scale of ICA is undetermined, apply normalization
    normCoef = max(abs(y), [], "all");
    y = y ./ normCoef;
    W = normCoef*W;
end
y = y.'; % time samples x sources
end

%--------------------------------------------------------------------------
function local_plotCost(cost)
    plot((0:size(cost, 1)-1), cost);
    set(gca, "FontName", "Arial", "FontSize", 14);
    xlabel("Number of iteration", "FontSize", 14);
    ylabel("Value of cost function", "FontSize", 14);
    grid on;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%