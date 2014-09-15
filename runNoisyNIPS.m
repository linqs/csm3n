%% Noisy NIPS experiment
%
% Variables:
%   nFold (def: 10)
%   noiseRate (def: .2)
%   runAlgos (def: [4 5 10])
%   inferFunc (def: UGM_Infer_TRBP)
%   decodeFunc (def: UGM_Decode_TRBP)
%   save2file (def: will not save)
%   makePlots (def: 0)

if ~exist('nFold','var')
	nFold = 10;
end
if ~exist('noiseRate','var')
	noiseRate = .2;
end
if ~exist('runAlgos','var')
	runAlgos = [4 5 10];
end
if ~exist('inferFunc','var')
	inferFunc = @UGM_Infer_TRBP;
end
if ~exist('decodeFunc','var')
	decodeFunc = @UGM_Decode_TRBP;
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

cd data/nips14;
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
[examples] = iidNoiseModel(nFold*(nTrain+nCV+nTest),2,noiseRate,2,.6,1,1,dataFig);
cd ../..;

expSetup = struct('foldIdx',foldIdx ...
				 ,'runAlgos',runAlgos ...
				 ,'decodeFunc',@UGM_Decode_TRBP,'inferFunc',@UGM_Infer_TRBP ...
				 ,'Cvec',[.00001 .00005 .0001 .0005 .001 .005 .01 .05 .1 1] ...
				 ,'Cvec2',0 ... %[.0001 .001 .01 .1 1 10] ...
				 ,'stepSizeVec',[.1 .2 .5 1 2 5 10] ...
				 ,'kappaVec',[.1 .2 .5 1 2 5 10] ...
				 ,'computeBaseline',1 ...
				 );
expSetup.optSGD = struct('maxIter',100 ...
						,'plotObj',objFig,'plotRefresh',100 ...
						,'verbose',0,'returnBest',1);
expSetup.optLBFGS = struct('Display','off','verbose',0 ...
						  ,'MaxIter',100,'MaxFunEvals',100);
expSetup.plotPred = predFig;

if exist('save2file','var')
	expSetup.save2file = save2file;
end

experiment;
