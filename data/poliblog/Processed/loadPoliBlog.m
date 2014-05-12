function [examples,foldIdx] = loadPoliBlog()
%
% Loads and formats the Political Blog data
%

% Load 4 examples
fprintf('Reading Feb1 ... ');
examples{1} = readExample('Feb_1',1,1);
fprintf('done.\n');
fprintf('Reading Feb2 ... ');
examples{2} = readExample('Feb_2',1,1);
fprintf('done.\n');
fprintf('Reading May1 ... ');
examples{3} = readExample('May_1',1,1);
fprintf('done.\n');
fprintf('Reading May2 ... ');
examples{4} = readExample('May_2',1,1);
fprintf('done.\n');

% Generate all permutations for 24 folds of size 4
idx = perms(1:4);
for fold = 1:size(idx,1)
	foldIdx(fold).tridx = idx(fold,1);
	foldIdx(fold).ulidx = idx(fold,2);
	foldIdx(fold).cvidx = idx(fold,3);
	foldIdx(fold).teidx = idx(fold,4);
end

