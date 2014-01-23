function examples = loadActivity(anno,feat,actions,compEdgeDist)
% 
% Loads an Activity Detection dataset.
%

if nargin < 4
	compEdgeDist = 0;
end

nState = length(actions);

% each sequence is an example
examples = cell(length(anno),1);

% loop over sequences
for s = 1:length(anno)
	
	% node index iterator
	nidx = 0;
	
	% node (index,id) map
	nodeId = [];
	
	% adjacencies
	adj = [];
	
	% labels/features
	y = [];
	Xnode = [];
	
	% loop over frames
	prevFrame = [];
	for f = 1:length(anno{s})
		
		% loop over nodes
		curFrame = [];
		for n = 1:length(anno{s}{f})
			
			% make sure action is in set of states
			a = find(actions == anno{s}{f}(n).act);
			if isempty(a)
				continue;
			end
			
			% node is in frame
			nidx = nidx + 1;
			curFrame(end+1) = nidx;
			
			% identity
			nodeId(nidx) = anno{s}{f}(n).id;

			% label
			y(nidx) = a;
			
			% node features (for now, jus ACDs)
			Xnode(1,:,nidx) = feat{s}{f}(n).acscore;
			
		end
		
		% nodes in current frame are adjacent
		for n1 = 1:length(curFrame)-1
			for n2 = n1+1:length(curFrame)
				adj(end+1,:) = [curFrame(n1) curFrame(n2)];
			end
		end
		
		% nodes with same identity are adjacent
		if f > 1
			for n1 = 1:length(prevFrame)
				for n2 = 1:length(curFrame)
					id1 = nodeId(prevFrame(n1));
					id2 = nodeId(prevFrame(n2));
					if id1 == id2
						adj(end+1,:) = [prevFrame(n1) curFrame(n2)];
						break;
					end
				end
			end
		end
		
		% previous frame becomes current frame
		prevFrame = curFrame;
	end
	
	% make example
	G = sparse(adj(:,1),adj(:,2),ones(size(adj,1),1),nidx,nidx);
	G = G + G';
	edgeStruct = UGM_makeEdgeStruct(G,nState,1);
	if compEdgeDist
		edgeStruct.edgeDist = makeEdgeDistribution(edgeStruct);
	end
	[Aeq,beq] = pairwiseConstraints(edgeStruct);
	Xedge = makeEdgeFeatures(Xnode,nodeId,edgeStruct.edgeEnds);
	examples{s} = makeExample(Xnode,Xedge,y',nState,edgeStruct,Aeq,beq);
	
end


