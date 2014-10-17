%% GrabCut experiment
%
% Variables:
%   runAlgos (def: [4 7])
%   inferFunc (def: UGM_Infer_CountBP)
%   decodeFunc (def: UGM_Decode_LBP)
%   Cvec (def: 10.^[-4:2])
%   save2file (def: will not save)

clear;
nEx = 50;
nTr = 5;
nCV = 5;
maxEvals = 200;

if ~exist('runAlgos','var')
	runAlgos = [4 7];
end
if ~exist('inferFunc','var')
	inferFunc = @UGM_Infer_CountBP;
end
if ~exist('decodeFunc','var')
	decodeFunc = @UGM_Decode_LBP;
end
if ~exist('Cvec','var')
	Cvec = 10.^[-4:2];
end

if any(runAlgos == 2) || any(runAlgos == 3)
	stepSizeVec = [.1 .2 .5 1 2];
else
	stepSizeVec = 1;
end

if any(runAlgos == 5)
	kappaVec = [.1 .2 .5 1 2 5 10];
else
	kappaVec = 1;
end

numFolds = 10;

%% Partition folds

rng(917);
% [~, shuffleOrder] = sort(rand(nEx,1)); % shuffle
shuffleOrder = 1:50; % don't shuffle

for i = 1:numFolds
    % disjoint train and validation
%     foldIdx(i).tridx = (i-1) * (nTr+nCV) + [1:nTr];
%     foldIdx(i).ulidx = [];
%     foldIdx(i).cvidx = mod((i-1) * (nTr+nCV) + [1:nTr] + nTr - 1, nEx) + 1;
%     foldIdx(i).teidx = setdiff(1:nEx, [foldIdx(i).tridx, foldIdx(i).cvidx]);

    % rotate train and validation round robin style    
    foldIdx(i).tridx = (i-1) * nTr + [1:nTr];
    foldIdx(i).ulidx = [];
    foldIdx(i).cvidx = mod((i-1) * nTr + [1:nTr] + nTr - 1, nEx) + 1;
    foldIdx(i).teidx = setdiff(1:nEx, [foldIdx(i).tridx, foldIdx(i).cvidx]);

    % shuffle
    foldIdx(i).tridx = shuffleOrder(foldIdx(i).tridx);
    foldIdx(i).cvidx = shuffleOrder(foldIdx(i).cvidx);
    foldIdx(i).teidx = shuffleOrder(foldIdx(i).teidx);
end

testFoldIdx(foldIdx);


%% Load data
cd data/grabcut;
makeEdgeDist = 1;
countBP = 2;
kappa = 1;
scaled = 1;
[examples] = loadGrabCut(nEx,makeEdgeDist,countBP,kappa,scaled);
cd ../../;

%% Experiment

expSetup = struct(...
	 'nFold',numFolds ...
    ,'foldIdx', foldIdx ...
    ,'runAlgos',runAlgos...
    ,'decodeFunc',decodeFunc ...
    ,'inferFunc',inferFunc ...
    ,'Cvec',Cvec ...
    ,'stepSizeVec',stepSizeVec ...
	,'kappaVec', kappaVec ...
    );

figure(3);
expSetup.optSGD = struct(...
	 'maxIter',maxEvals ...
	,'verbose',1,'returnBest',1 ...
	,'plotObj', gcf, 'plotRefresh', 5);
expSetup.optLBFGS = struct(...
	 'Display','iter' ...
    ,'MaxIter',maxEvals ...
    ,'MaxFunEvals',maxEvals ...
    ,'plotObj', gcf ...
    ,'plotRefresh', 5 ...
    ,'verbose', 1 ...
    );

algoNames = {'MLE','PERC','M3N','M3NFW','SCTSM','VCTSM','VCTSMlog'};

algoString = '';
for i = 1:length(runAlgos)
    algoString = [algoString algoNames{runAlgos(i)}];
end

if exist('save2file','var')
	expSetup.save2file = save2file;
end
% expSetup.save2file = sprintf('results/grabCutResults_%s_%d_%d', algoString, nEx, numFolds);

%expSetup.plotFunc = @plotGrabCut;

%figure(1);
%clf;
%ax{1} = subplot(121);
%ax{2} = subplot(122);

%expSetup.plotFuncAxis = ax;

experiment;
