function [examples,foldIdx] = loadPoliBlog(makeEdgeDist,makeCounts)
%
% Loads and formats the Political Blog data
%

if nargin < 1
	makeEdgeDist = 0;
end

% Load 4 examples
for i = 1:4
	fprintf('Reading split %d ... ',i);
	examples{i} = readExample(sprintf('split%d',i),makeEdgeDist,makeCounts);
	fprintf('done.\n');
end

% Generate all permutations for 24 folds of size 4
idx = perms(1:4);
for fold = 1:size(idx,1)
	foldIdx(fold).tridx = idx(fold,1);
	foldIdx(fold).ulidx = idx(fold,2);
	foldIdx(fold).cvidx = idx(fold,3);
	foldIdx(fold).teidx = idx(fold,4);
end

