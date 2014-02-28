function [f, g] = vctsmObj(x, examples, C, kappa, varargin)

% Outputs the objective value and gradient of the VCTSM learning objective
% using the dual of loss-augmented inference to make the objective a
% minimization.
%
% x : current point in optimization
% examples : nEx x 1 cell array of examples, each containing:
%	oc : full overcomplete vector representation of Y
%		 (including high-order terms)
%	ocLocalScope : number of local terms in oc
%	Aeq : nCon x length(oc) constraint A matrix
%	beq : nCon x 1 constraint b vector
%	Fx : nParam x length(oc) feature map
%	suffStat : nParam x 1 vector of sufficient statistics (i.e., Fx * oc)
% C : regularization constant or vector
% kappa : predefined modulus of convexity
% varargin : optional arguments (required by minFunc)

nEx = length(examples);
nParam = max(examples{1}.edgeMap(:));

% parse current position
w = x(1:nParam);
lambda = x((nParam+1):end);

% for numerical stability, log kappa
logkappa = log(kappa);

% init outputs
% f = 0.5 * exp(-2*logkappa) * (C .* w)' * w;
f = 0.5 * (C .* w)' * w;
if nargout == 2
% 	gradW = exp(-2*logkappa) * (C .* w);
	gradW = (C .* w);
	gradLambda = zeros(length(lambda),1);
end

% main loop
iter = 1;
for i = 1:nEx
	
	% static variables for example i
	mu = examples{i}.oc;
	A = examples{i}.Aeq;
	b = examples{i}.beq;
	Fx = examples{i}.Fx;
	ss = examples{i}.suffStat;
	nLoc = examples{i}.ocLocalScope;
	[nCon,nAll] = size(A);
	
	% intermediate variables
	lam = lambda(iter:iter+nCon-1);
	ell = zeros(nAll,1);
	ell(1:nLoc) = 1 - 2*mu(1:nLoc);
	z = (Fx'*w + ell + A'*lam);
	y = exp(exp(-logkappa)*z - 1);
	
	% objective
	loss = exp(logkappa)*sum(y) - w'*ss - lam'*b + sum(mu(1:nLoc));
	f = f + loss;
	
	% gradient
	if nargout == 2
		gradW = gradW + Fx * y - ss;
		gradLambda(iter:iter+nCon-1) = A * y - b;
	end
	
	iter = iter + nCon;
	
end

if nargout == 2
	g = [gradW; gradLambda];
end


