%cd ~/Dropbox/Research/csm3n

%setUpPath;

clear;
nEx = 50;
nTr = 5;
nCV = 5;
maxEvals = 200;

% m3n
%countBP = false;
%runAlgos = 10;
%
% countbp
% countBP = true;
% runAlgos = 4;
%
% % trbp
% countBP = false;
% runAlgos = 4;

countBP = false;
runAlgos = [4 10];

numFolds = 10;

rng(1);
[~, shuffleOrder] = sort(rand(nEx,1));
% shuffleOrder
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

%% GrabCut experiment
cd data/grabcut;
[examples] = loadGrabCut(1, nEx, countBP, 1);
cd ../../;

%%
inferFunc = @UGM_Infer_TRBP;
%if countBP
%    inferFunc = @UGM_Infer_CountBP;
%end


expSetup = struct('nFold',numFolds ...
    , 'foldIdx', [foldIdx] ...
    , 'runAlgos',runAlgos...
    , 'decodeFunc',@UGM_Decode_TRBP ...
    , 'inferFunc',inferFunc ...
    , 'Cvec',10.^[-4:2] ...
    , 'Cvec2',[0.01 0.1 1.0] ...
    );


figure(3);
expSetup.optSGD = struct('maxIter',maxEvals ...
    ,'verbose',1,'returnBest',1, 'plotObj', gcf, 'plotRefresh', 5);
expSetup.optLBFGS = struct('Display','iter' ...
    ,'MaxIter',maxEvals ...
    ,'MaxFunEvals',maxEvals ...
    , 'plotObj', gcf ...
    , 'plotRefresh', 5 ...
    , 'verbose', 1 ...
    );

algoNames = {'MLE','PERC','M3N','M3NFW','SCTSM','VCTSM','VCTSMlog'};

algoString = '';
for i = 1:length(runAlgos)
    algoString = [algoString algoNames{runAlgos(i)}];
end
expSetup.save2file = sprintf('results/grabCutResults_%s_%d_%d', algoString, nEx, numFolds);

%expSetup.plotFunc = @plotGrabCut;

%figure(1);
%clf;
%ax{1} = subplot(121);
%ax{2} = subplot(122);

%expSetup.plotFuncAxis = ax;


experiment;
