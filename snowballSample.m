function [subgraphs] = snowballSample(A, k, jumpRate, seeds, makePlots)
%
% Partitions graph in k subgraphs using snowball sampling.
%

N = size(A,1);

if ~exist('jumpRate','var') || isempty(jumpRate)
    jumpRate = 0;
end
if ~exist('seeds','var') || isempty(seeds)
    seeds = randi(N,k,1);
end
if ~exist('makePlots','var') || isempty(makePlots)
	makePlots = 0;
end

% Symmetrize adjacency matrix (just in case)
A = A | A';

% Init selected and remaining indices
selected = zeros(N,k);
remaining = ones(N,1);
for p = 1:k
	selected(seeds(p),p) = 1;
	remaining(seeds(p)) = 0;
end

% Init frontiers
frontiers = zeros(N,k);
for p = 1:k
	frontiers(:,p) = A(:,seeds(p)) & remaining;
end

% Partition nodes
next = seeds;
while nnz(remaining) > 0
	for p = 1:k
		% Update frontier for changes to other partitions
		frontiers(:,p) = frontiers(:,p) & remaining;
		
		% Choose set from which to select next random node
		if rand() < jumpRate || all(~frontiers(:,p))
			I = find(remaining);
		else
			I = find(frontiers(:,p));
		end
		next(p) = I(randi(length(I)));
		
		% Update indices and remaining
		selected(next(p),p) = 1;
		remaining(next(p)) = 0;
		if nnz(remaining) == 0
			break
		end
		
		% Update frontier for current changes
		frontiers(:,p) = (frontiers(:,p) | A(:,next(p))) & remaining;
	end
end

% Create subgraphs
for p = 1:k
	subgraphs(p).nodes = find(selected(:,p));
	subgraphs(p).A = A(subgraphs(p).nodes,subgraphs(p).nodes);
end

% Plots for testing
if makePlots
	B = zeros(N);
	figure();
	for p = 1:k
		subplot(1,k,p);
		spy(subgraphs(p).A);
		[I,J] = find(subgraphs(p).A);
		idx = sub2ind([N N],subgraphs(p).nodes(I),subgraphs(p).nodes(J));
		B(idx) = 1;
	end
	figure();
	subplot(1,2,1);
	spy(A);
	subplot(1,2,2);
	spy(B);
end
