function [f, g] = m3nObj(x, examples, C, decodeFunc, varargin)
%
% Outputs the objective value and gradient of the M3N learning objective
% (i.e., VCTSM with a fixed kappa).
%
% x : current point in optimization
% examples : nEx x 1 cell array of examples, each containing:
%	Fx : nParam x length(oc) feature map
%	suffStat : nParam x 1 vector of sufficient statistics (i.e., Fx * oc)
%	Ynode : nState x nNode overcomplete matrix representation of labels
% C : regularization constant or vector
% decodeFunc : decoder function
% kappa : predefined modulus of convexity
% varargin : optional arguments (required by minFunc)

nEx = length(examples);
nParam = max(examples{1}.edgeMap(:));

% Parse current position
w = x(1:nParam);

% Init outputs
f = 0.5 * (C .* w)' * w;
if nargout == 2
	gradW = (C .* w);
end

% Main loop
for i = 1:nEx
	
	% Grab ith example.
	ex = examples{i};
	Fx = ex.Fx;
	ss_y = ex.suffStat;
	Ynode = ex.Ynode'; % assumes Ynode is (nState x nNode)
	
	% Loss-augmented inference
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
	yMAP = decodeFunc(nodePot.*exp(1-Ynode),edgePot,ex.edgeStruct,varargin{:});

	% Compute sufficient statistics
	nodeBel = zeros(size(nodePot));
	edgeBel = zeros(size(edgePot));
	for n = 1:ex.nNode
		nodeBel(n,yMAP(n)) = 1;
	end
	for e = 1:ex.nEdge
		n1 = ex.edgeStruct.edgeEnds(e,1);
		n2 = ex.edgeStruct.edgeEnds(e,2);
		edgeBel(yMAP(n1),yMAP(n2),e) = 1;
	end
	ss_mu = Fx * [reshape(nodeBel',[],1) ; edgeBel(:)];
	
	% Difference of sufficient statistics
	ssDiff = ss_mu - ss_y;
	
	% Objective
	L1 = norm(Ynode(:)-nodeBel(:), 1);
	loss = (w'*ssDiff + 0.5*L1) / (nEx*ex.nNode);
	f = f + loss;
	
	% Gradient
	if nargout == 2
		gradW = gradW + ssDiff / (nEx*ex.nNode);
	end
	
end

if nargout == 2
	g = gradW;
end


