cd ~/Dropbox/Research/csm3n

setUpPath;

clear;
nEx = 50;
nTr = 1;
nCV = 1;

numFolds = 50;

for i = 1:numFolds
    foldIdx(i).tridx = (i-1) * nTr + [1:nTr];
    foldIdx(i).ulidx = [];
    foldIdx(i).cvidx = mod(i * nTr + [0:nCV-1], nEx) + 1;
    foldIdx(i).teidx = setdiff(1:nEx, [foldIdx(i).tridx, foldIdx(i).cvidx]);
end

%% GrabCut experiment
cd ~/Dropbox/Research/csm3n/data/grabcut;
[examples] = loadGrabCut(1, nEx);
cd ../../;

%%

expSetup = struct('nFold',numFolds,...
    'foldIdx', [foldIdx],...
    'runAlgos',[2 4],...
    'decodeFunc',@UGM_Decode_TRBP,...
    'inferFunc',@UGM_Infer_TRBP,...
    'Cvec',[0.0001 .001 .01 .1 1 10],...
    'stepSizeVec',[.02]);

expSetup.optSGD = struct('maxIter',500 ...
						,'verbose',0,'returnBest',1);
expSetup.optLBFGS = struct('Display','iter', 'verbose', 0);

% expSetup.save2file = 'grabCutResults_adhoc';
expSetup.save2file = 'grabCutResults_fixed_50';

expSetup.plotFunc = @plotGrabCut;

figure(1);
clf;
expSetup.plotFuncAxis = gca;

experiment;
