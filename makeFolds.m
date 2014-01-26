function foldIdx = makeFolds(nEx,nFold,varargin)
% 
% Automatically generates folds indices.
% 

if length(varargin) == 1 && length(varargin{1}) == 4
	nTrain = varargin{1}(1);
	nUnlab = varargin{1}(2);
	nCV = varargin{1}(3);
	nTest = varargin{1}(4);
elseif length(varargin) == 4
	nTrain = varargin{1};
	nUnlab = varargin{2};
	nCV = varargin{3};
	nTest = varargin{4};
else
	error('USAGE: makeFolds(nEx,nFold,[nTrain nUnlab nCV nTest]) or makeFolds(nEx,nFold,nTrain, nUnlab, nCV, nTest)');
end

nExFold = nTrain + nUnlab + nCV + nTest;
assert(nExFold <= nEx, 'Number of examples per fold greater than examples.');

for fold = 1:nFold
	
	if (fold * nExFold) > nEx
		warning('Number of folds exceeds available examples.')
		break;
	end
	
	fidx = (fold-1) * nExFold;
	foldIdx(fold).tridx = fidx+1:fidx+nTrain;
	foldIdx(fold).ulidx = fidx+nTrain+1:fidx+nTrain+nUnlab;
	foldIdx(fold).cvidx = fidx+nTrain+nUnlab+1:fidx+nTrain+nUnlab+nCV;
	foldIdx(fold).teidx = fidx+nTrain+nUnlab+nCV+1:fidx+nTrain+nUnlab+nCV++nTest;
	
end

