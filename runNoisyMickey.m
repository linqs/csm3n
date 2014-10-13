%% Noisy Mickey experiment
%
% Variables:
%   nFold (def: 10)
%   noiseRate (def: .2)
%   runAlgos (def: [4 5 7])
%   inferFunc (def: UGM_Infer_TRBP)
%   decodeFunc (def: UGM_Decode_TRBP)
%	Cvec (def: [.0001 .0005 .001 .0025 .005 .0075 .01 .025 .05 .1])
%   kappaVec (def: [.1 .2 .5 .75 1 1.5 2 5])
%   save2file (def: will not save)
%   makePlots (def: 0)

if ~exist('nFold','var')
	nFold = 10;
end
if ~exist('noiseRate','var')
	noiseRate = .2;
end
if ~exist('runAlgos','var')
	runAlgos = [4 5 7];
end
if ~exist('inferFunc','var')
	inferFunc = @UGM_Infer_TRBP;
end
if ~exist('decodeFunc','var')
	decodeFunc = @UGM_Decode_TRBP;
end
if ~exist('Cvec','var')
	Cvec = [.0001 .0005 .001 .0025 .005 .0075 .01 .025 .05 .1];
end
if ~exist('kappaVec','var')
	kappaVec = [.1 .2 .5 .75 1 1.5 2 5];
end
if ~exist('makePlots','var')
	makePlots = 0;
end


if makePlots
	dataFig = 101;
	objFig = 102;
	predFig = 103;
else
	dataFig = 0;
	objFig = 0;
	predFig = 0;
end

% seed the RNG
rng(0);

nTrain = 1;
nCV = 1;
nTest = 10;
for f = 1:nFold
	sidx = (f-1)*(nTrain+nCV+nTest);
	foldIdx(f).tridx = sidx+1:sidx+nTrain;
	foldIdx(f).ulidx = [];
	foldIdx(f).cvidx = sidx+nTrain+1:sidx+nTrain+nCV;
	foldIdx(f).teidx = sidx+nTrain+nCV+1:sidx+nTrain+nCV+nTest;
end

cd data/mickey;
nFeat = 10; % 2;
noiseType = 1; % 2;
scale = .6;
[examples] = iidNoiseModel(nFold*(nTrain+nCV+nTest),nFeat,noiseRate,noiseType,scale,1,1,dataFig);
cd ../..;

expSetup = struct('foldIdx',foldIdx ...
				 ,'runAlgos',runAlgos ...
				 ,'decodeFunc',decodeFunc,'inferFunc',inferFunc ...
				 ,'Cvec',Cvec ...
				 ,'kappaVec',kappaVec ...
				 ,'computeBaseline',1 ...
				 );
expSetup.optSGD = struct('maxIter',200 ...
						,'plotObj',objFig,'plotRefresh',10 ...
						,'verbose',1,'returnBest',1);
expSetup.optLBFGS = struct('Display','iter','verbose',0 ...
						  ,'MaxIter',200,'MaxFunEvals',200);
expSetup.plotPred = predFig;

if exist('save2file','var')
	expSetup.save2file = save2file;
end

experiment;
