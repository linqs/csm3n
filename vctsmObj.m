function [f, g] = vctsmObj(examples, C, varargin)

% Outputs the objective value and gradient of the VCTSM learning objective
% using the dual of loss-augmented inference to make the objective a
% minimization.
%
% examples : nEx x 1 cell array of examples, each containing:
%	oc : full overcomplete vector representation of Y
%		 (including high-order terms)
%	ocLocalScope : number of local terms in oc
%	Aeq : nCon x length(oc) constraint A matrix
%	beq : nCon x 1 constraint b vector
%	F : nParam x length(oc) feature map
%	suffStat : nParam x 1 vector of sufficient statistics (i.e., F * oc)
% C : regularization constant or vector
% varargin : optional arguments (required by minFunc)

nEx = length(examples);
nParam = double(max(examples{1}.edgeMap));

% parse current position
w = x(1:nParam);
logkappa = x(nParam+1);
lambda = x((nParam+2):end);

% loop variables
iter = 1;
f = 0.5 * exp(-2*logkappa) * (C .* w)' * w;
if nargout == 2
	gradW = exp(-2*logkappa) * (C .* w);
	gradKappa = -exp(-2*logkappa) * (C .* w)' * w;
	gradLambda = zeros(length(lambda));
end

for i = 1:nEx
	
	% static variables for example i
	mu = examples{i}.oc;
	A = examples{i}.Aeq;
	b = examples{i}.beq;
	F = examples{i}.F;
	ss = examples{i}.suffStat;
	nLoc = examples{i}.ocLocalScope;
	[nCon,nAll] = size(A);
	
	% intermediate variables
	lam = lambda(iter:iter+nCon-1);
	ell = zeros(nAll,1);
	ell(1:nLoc) = 1 - 2*mu(1:nLoc);
	z = (F'*w + ell + A'*lam);
	y = exp( exp(-logkappa)*z - 1 );
	
	% objective
	loss = exp(logkappa)*sum(y) - w'*ss - lam'*b + sum(mu(1:nLoc));
	f = f + loss;
	
	% gradient
	if nargout == 2
		gradW = gradW + F * y - ss;
		gradKappa = gradKappa + y' * (exp(logkappa) - z);
		gradLambda(iter:iter+nCon-1) = A * y - b;
	end
	
	iter = iter + nCon;
	
end

if nargout == 2
	g = [gradW; gradKappa; gradLambda];
end


% OLD CODE

% A = S.Aeq;
% b = S.beq;
% 
% delta = sum(labels(scope));
% ell = zeros(size(labels));
% ell(scope) = 1-2*labels(scope);
% wtw = w' * w;
% z = (F'*w + ell + A'*lambda);
% y = exp( exp(-logkappa)*z - 1 );
% 
% loss = exp(logkappa)*sum(y) - w'*F_labels - lambda'*b;
% 
% f = 0.5 * exp(-2*logkappa) * C * wtw + loss + delta;
% 
% if nargout == 2
%     gradW = exp(-2*logkappa) * C * w + F * y - F_labels;
%     gradKappa = -exp(-2*logkappa) * C * wtw +  y' * (exp(logkappa) - z);
%     gradLambda = A * y - b;
%     g = [gradW; gradKappa; gradLambda];
% end
