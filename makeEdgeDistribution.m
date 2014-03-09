function mu = makeEdgeDistribution(edgeStruct,type)
%
% Computes the edge distribution for TRBP.
%

if nargin < 2
	type = 1;
end

nNode = edgeStruct.nNodes;
nEdge = edgeStruct.nEdges;

switch(type)
	
	case 0
		% Ordinary BP (not a valid distribution over trees, so not convex)
		mu = ones(nEdge,1);
		
	case 1
		% Generate Random Spanning Trees until all edges are covered
		edgeEnds = edgeStruct.edgeEnds;
		i = 0;
		edgeAppears = zeros(nEdge,1);
		while 1
			i = i+1
			edgeAppears = edgeAppears+minSpan(nNode,[edgeEnds rand(nEdge,1)]);
			if all(edgeAppears > 0)
				break;
			end
			find(edgeAppears == 0)
		end
		mu = edgeAppears/i;
		
	case 2
		% Compute all spanning trees of the dense graph (not a valid distribution over trees for over graphs)
		mu = ((nNode-1)/nEdge) * ones(nEdge,1);
		
end


