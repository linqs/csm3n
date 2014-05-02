%% Noisy NIPS experiment
%
% Variables:
%   nFold (def: 10)
%   noiseRate (def: .2)
%   runAlgos (def: [2 4 5])
%   inferFunc (def: UGM_Infer_TRBP)
%   decodeFunc (def: UGM_Decode_TRBP)
%   filename (def: will not save)

if ~exist('nFold','var')
	nFold = 10;
end
if ~exist('noiseRate','var')
	noiseRate = .2;
end
if ~exist('runAlgos','var')
	runAlgos = [2 4 5];
end
if ~exist('inferFunc','var')
	inferFunc = @UGM_Infer_TRBP;
end
if ~exist('decodeFunc','var')
	decodeFunc = @UGM_Decode_TRBP;
end

cd data/nips14;
nTrain = 1;
nCV = 1;
nTest = 10;
[examples] = loadExamples(nFold*(nTrain+nCV+nTest),noiseRate,.5,1,1,101);
for f = 1:nFold
	sidx = (f-1)*(nTrain+nTest);
	foldIdx(f).tridx = sidx+1:sidx+nTrain;
	foldIdx(f).ulidx = [];
	foldIdx(f).cvidx = sidx+nTrain+1:sidx+nTrain+nCV;
	foldIdx(f).teidx = sidx+nTrain+nCV+1:sidx+nTrain+nCV+nTest;
end
cd ../..;

expSetup = struct('foldIdx',foldIdx ...
				 ,'runAlgos',[2 4 5] ...
				 ,'decodeFunc',@UGM_Decode_TRBP,'inferFunc',@UGM_Infer_TRBP ...
				 ,'Cvec',[.001 .01 .1 1 10 100 1000] ...
				 ,'stepSizeVec',[.02 .04 .06] ...
				 ,'kappaVec',[.1 .2 .5 1 2 5 10 20 50 100] ...
				 );
expSetup.optSGD = struct('maxIter',1000 ...
						,'plotObj',102,'plotRefresh',100 ...
						,'verbose',0,'returnBest',1);
expSetup.optLBFGS = struct('Display','iter');
expSetup.plotPred = 103;
experiment

experiment;
