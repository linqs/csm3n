function [w,f] = bcfw(examples, decodeFunc, lambda, options, w, varargin)
%
% Trains an MRF using max-margin block coordinate frank-wolfe.
%
% examples : nEx x 1 cell array of examples, each containing:
%	Fx : nParam x length(oc) feature map
%	suffStat : nParam x 1 vector of sufficient statistics (i.e., Fx * oc)
%	Ynode : nState x nNode overcomplete matrix representation of labels
% decodeFunc : decoder function
% C : optional regularization constant or vector (def: 1)
% options : optional struct of optimization options for subgradient descent:
% 			maxIter : iterations (def: 100*length(examples))
% 			stepSize : step size (def: 1)
% 			verbose : verbose mode (def: 0)
% w : init weights (optional: def=zeros)

% parse input
assert(nargin >= 2, 'USAGE: bcfw(examples,decodeFunc)')
if nargin < 3
    lambda = 1;
end
if nargin < 4 || ~isstruct(options)
    options = struct();
end
if ~isfield(options,'maxIter')
    options.maxIter = 100 * length(examples);
end
if ~isfield(options,'verbose')
    options.verbose = 0;
end
if ~isfield(options,'tolerance')
    options.tolerance = 1e-6;
end
if ~isfield(options,'plotObj')
    options.plotObj = false;
end
if ~isfield(options,'plotRefresh')
    options.plotRefresh = 10;
end
if nargin < 5 || isempty(w)
    nParam = max(examples{1}.edgeMap(:));
    w = zeros(nParam,1);
end

wi = 0;
wavg = w;
l = 0;
li = 0;


if options.plotObj
    figure(options.plotObj);
    clf;
    subplot(311);
    objAx = gca;
    subplot(312);
    gapAx = gca;
    fvec = m3nObj(w, examples,lambda,decodeFunc,varargin{:});
    normX = norm(w);
    subplot(313);
    dualAx = gca;
    dualObj = [];
end

N = length(examples);

for k = 1:options.maxIter
    i = randi(N);
    
    ex = examples{i};
    Fx = ex.Fx;
    ss_y = ex.suffStat;
    Ynode = ex.Ynode; % assumes Ynode is (nState x nNode)
    
    % Loss-augmented inference
    [nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
    yMAP = decodeFunc(nodePot.*exp(1-Ynode'),edgePot,ex.edgeStruct,varargin{:});
    
    % Compute sufficient statistics
    ocrep = overcompletePairwise(yMAP,ex.nState,ex.edgeStruct);
    
    ss_mu = Fx * ocrep;
    
    % Difference of sufficient statistics
    ssDiff = (ss_y - ss_mu) / ex.nNode; % note this is backwards from the m3n version
    
    % Objective
    L1 = 0.5 * norm(Ynode(:)-ocrep(1:ex.ocLocalScope), 1) / ex.nNode;
    
    ws = (1/(N*lambda)) * ssDiff;
    ls = (1/N) *  L1;
    
    gap(k) = lambda * (w - ws)'*w - l + ls;
    % compute dual objective
    dualObj(k) = 0.5 * lambda * norm(w)^2 + l;
    
    
    % compute step size
    
    wimws = wi-ws;
    gamma = (lambda * (wimws)'*w - li + ls) / (lambda * (wimws'*wimws));
    gamma = max(0, min(1, gamma));
    
    % take step
    
    wiPrev = wi;
    liPrev = li;
    
    wi = (1-gamma) * wi + gamma * ws;
    li = (1-gamma) * li + gamma * ls;
    
    w = w + wi - wiPrev;
    l = l + li - liPrev;
    
    wavg = k/(k+2) * wavg + 2/(k+2) * w;
    
    if options.plotObj || (gap(k) < options.tolerance && N == 1)
        
        if mod(k,options.plotRefresh) == 0
            fvec(end+1) = m3nObj(w, examples,lambda,decodeFunc,varargin{:});
            normX(end+1) = norm(w);
            hAx = plotyy(1:length(fvec),fvec, 1:length(normX),normX, 'Parent', objAx);
            ylabel(hAx(1),'Objective'); ylabel(hAx(2),'norm(x)');
            
            semilogy(1:k, max(gap, 0), 'Parent', gapAx);
            xlabel(gapAx, 'Iteration');
            ylabel(gapAx, 'Duality Gap');
            drawnow;
            
            
            semilogy(1:length(dualObj), dualObj, 'Parent', dualAx);
            ylabel('Dual Objective');
        end
    end
    
    if gap(k) < options.tolerance && N == 1
        break;
    end
end

if nargout == 2
    f = m3nObj(w, examples,lambda,decodeFunc,varargin{:});
end

if k == options.maxIter
    if options.verbose && N == 1
        fprintf('Frank-Wolfe did not converge in %d iterations. Using average w\n', k);
    end
    % use average w because it has better convergence properties if we
    % never achieved a certificate of convergence
    w = wavg;
else
    fprintf('Frank-Wolfe reached certificate of optimality within tolerance %s in %d iterations\n', ...
        options.tolerance, k);
end


% checkObjective(@(x) m3nObj(x,examples,lambda,decodeFunc,varargin{:}), w, 10.0)

% w = wavg;
