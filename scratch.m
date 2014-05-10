%% Test stabilityObj and csm3nObj
clear;
examples = noisyX(4,1,0,1,0);
decoder = @UGM_Decode_LBP;
expSetup = struct('nFold',1,'foldDist',[1 1 1 1],'runAlgos',2,'Cvec',100,'decoder',decoder);
experiment;
w = w + 4*randn(size(w));
options = struct('verbose',1,'plotObj',1);
for i = 1:length(examples)
	ex = examples{i};
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
	y = decoder(nodePot,edgePot,ex.edgeStruct);
	[f,g,x_p] = stabilityObj(w,ex,y,decoder,options);
	fprintf('Stability objective = %f\n', f);
% 	fprintf('First 20 entries of x,x''\n');
% 	x = reshape(squeeze(ex.Xnode(1,:,:)),[],1);
% 	disp([x(1:20) x_p(1:20)]);
% 	fprintf('Num. permutations >= .50: %d\n', nnz(abs(x-x_p)>=.5));
% 	fprintf('Num. permutations >= .25: %d\n', nnz(abs(x-x_p)>=.25));
% 	fprintf('Num. permutations >= .10: %d\n', nnz(abs(x-x_p)>=.1));
% 	fprintf('Num. permutations >= .01: %d\n', nnz(abs(x-x_p)>=.01));
end

[f,g] = csm3nObj(w,examples(1),examples(end),decoder,0,.25,options);
fprintf('CSM3N objective = %f\n', f);

[f,g] = caccObj(w,examples(1),decoder,100,options);
fprintf('CACC objective = %f\n', f);


%% Derivative check for stabilityObj (note: uncomment lines in stabilityObj)
clear;
examples = noisyX(20,1,0,1,0);
experiment;
wnoisy = randn(size(w));
for i = 1:length(examples)
	stabilityObj(w,examples{i},@UGM_Decode_LBP);
	stabilityObj(wnoisy,examples{i},@UGM_Decode_LBP);
end

for i = 1:length(examples)
	stabObj = @(x,varargin) stabilityObj(x,examples{i},@UGM_Decode_LBP,varargin{:});
	fastDerivativeCheck(stabObj,w);
	for j = 1:3
		wnoisy = w + randn(size(w));
		fastDerivativeCheck(stabObj,wnoisy);
	end
end


%% Test CSM3N
clear;
examples = noisyX(10,1,0,1,0);
ex_l = examples(1:5);
ex_u = examples(6:10);
[w,fAvg] = trainCSM3N(ex_l,ex_u,@UGM_Decode_LBP,100)


%% SCTSM vs VCTSM
clear;
nFold = 4;
examples = noisyX(4*nFold,1,0,1,0);
Xdesc = struct('discreteX',1,'nonneg',1);
expSetup = struct('Xdesc',Xdesc,...
				  'nFold',nFold,'foldDist',[1 1 1 1],...
				  'runAlgos',[4 5],...
				  'Cvec',[0.1 1 100],...
				  'nStabSamp',0,...
				  'decodeFunc',@UGM_Decode_TRBP,'inferFunc',@UGM_Decode_TRBP);
% expSetup.optSGD = struct('maxIter',200);


%% Discrete noisyX
clear;
nFold = 1;
cd data
examples = noisyX(7*nFold,2,0,1,1);
% examples = noisyX(7*nFold,.3,0,1,1);
cd ..;
Xdesc = struct('discreteX',1,'nonneg',1);
expSetup = struct('Xdesc',Xdesc,...
				  'nFold',nFold,'foldDist',[1 0 1 5],...
				  'runAlgos',[2 4 5],...
				  'Cvec',1,...
				  'kappaVec',[.1 .2 .5 1 2],...
				  'nStabSamp',0,...
				  'decodeFunc',@UGM_Decode_TRBP,'inferFunc',@UGM_Infer_TRBP);
% expSetup.optSGD = struct('maxIter',200);


%% Continuous noisyX
clear;
nFold = 1;
cd data
examples = noisyX(7*nFold,2,0,0,1);
cd ..
Xdesc = struct('discreteX',0,'nonneg',0);
expSetup = struct('Xdesc',Xdesc,...
				  'nFold',nFold,'foldDist',[1 0 1 5],...
				  'runAlgos',[2 4 5],...
				  'Cvec',1,...
				  'kappaVec',[.1 .2 .5 1 2],...
				  'nStabSamp',0,...
				  'decodeFunc',@UGM_Decode_TRBP,'inferFunc',@UGM_Infer_TRBP);
% expSetup.optSGD = struct('maxIter',200);


%% PoliBlog experiment
clear;
cd data/poliblog/Raw/pcaProcessed;
[examples,foldIdx] = loadPoliBlog(1,1);
% perturbed = perturbExamples(examples,.15);
cd ../../../../;
Xdesc = struct('discreteX',0,'nonneg',0);%struct('discreteX',1);
expSetup = struct('nFold',1,'foldDist',[1 0 1 2],'Xdesc',Xdesc,...
				  'runAlgos',[2 4],...
				  'decodeFunc',@UGM_Decode_TRBP,'inferFunc',@UGM_Infer_TRBP,...
				  'nStabSamp',0,...
				  'Cvec',1,'CvecRel',0);
% expSetup = struct('foldIdx',foldIdx,'Xdesc',Xdesc,...
% 				  'runAlgos',[1 2 4],...
% 				  'decodeFunc',@UGM_Decode_TRBP,'inferFunc',@UGM_Infer_TRBP,...
% 				  'nStabSamp',0,...
% 				  'Cvec',[.1 .5 1 5 10 50 100 500 1000 5000 10000],'CvecRel',0,...
% 				  'save2file','poliblog1.mat');
expSetup.optSGD = struct('maxIter',200,'stepSize',1e-4);


%% Cora experiment
clear;
cd data;
[examples,foldIdx] = loadDocData('cora/cora.mat',3,{1:750,751:1750,1751:2708},50,1);
% [examples] = loadDocDataSnowball('cora/cora.mat',5,20,0,[1 600 1200 1800 2400],1,1);
cd ..;

expSetup = struct('nFold',1,'foldDist',[1 0 1 1] ...
				 ,'runAlgos',4 ...
				 ,'decodeFunc',@UGM_Decode_TRBP,'inferFunc',@UGM_Infer_TRBP ...
				 ,'Cvec',.0001 ...
				 ,'stepSizeVec',1 ...
				 ,'kappaVec',[.1 1 10 100] ...
				 );
expSetup.optSGD = struct('maxIter',100 ...
						,'plotObj',103,'plotRefresh',10 ...
						,'verbose',1,'returnBest',1);
expSetup.optLBFGS = struct('Display','iter','verbose',3);

experiment


%% Weizmann 1obj experiment

clear;
load('data/weizmann/w1obj_1-20.mat');
nFold = 5;
Xdesc = struct('discreteX',0,'nonneg',0);
expSetup = struct('Xdesc',Xdesc,...
				  'nFold',nFold,'foldDist',[1 1 1 1],...
				  'runAlgos',[1 2 4],...
				  'Cvec',[100],...
				  'nStabSamp',0,...
				  'decodeFunc',@UGM_Decode_TRBP,'inferFunc',@UGM_Infer_TRBP);
% expSetup.optSGD = struct('maxIter',200);


%% Noisy Image

clear;
cd data/nips14;
nFold = 1;
examples = [];
nTrain = 1;
nTest = 10;
for f = 1:nFold
	[ex_high,ex_low] = conceptDrift(.5,10,nTrain,nTest,.2,5,10,0);
	examples = [examples ; ex_high(:) ; ex_low(:)];
	sidx = (f-1)*(nTrain+nTest);
	foldIdx(f).tridx = sidx+1:sidx+nTrain;
	foldIdx(f).ulidx = [];
	foldIdx(f).cvidx = foldIdx(f).tridx;
	foldIdx(f).teidx = sidx+nTrain+1:sidx+nTrain+nTest;
end
cd ../..;

clear;
cd data/nips14;
nState = 10;
w_ratio = .25;
alpha = .7;
bias = 1;
nFold = 1;
nTrain = 1;
nCV = 1;
nTest = 10;
examples = entropicModel(.6,nState,nFold*(nTrain+nCV+nTest),w_ratio,alpha,bias,[],101);
for f = 1:nFold
	sidx = (f-1)*(nTrain+nCV+nTest);
	foldIdx(f).tridx = sidx+1:sidx+nTrain;
	foldIdx(f).ulidx = [];
	foldIdx(f).cvidx = sidx+nTrain+1:sidx+nTrain+nCV;
	foldIdx(f).teidx = sidx+nTrain+nCV+1:sidx+nTrain+nCV+nTest;
end
cd ../..;

clear;
cd data/nips14;
nFold = 1;
nTrain = 1;
nCV = 1;
nTest = 10;
[examples] = iidNoiseModel(nFold*(nTrain+nCV+nTest),4,2,1,.6,1,0,101);
for f = 1:nFold
	sidx = (f-1)*(nTrain+nCV+nTest);
	foldIdx(f).tridx = sidx+1:sidx+nTrain;
	foldIdx(f).ulidx = [];
	foldIdx(f).cvidx = sidx+nTrain+1:sidx+nTrain+nCV;
	foldIdx(f).teidx = sidx+nTrain+nCV+1:sidx+nTrain+nCV+nTest;
end
cd ../..;

expSetup = struct('foldIdx',foldIdx ...
				 ,'runAlgos',[4 10] ...%[4 5 10 12] ...
				 ,'decodeFunc',@UGM_Decode_LBP,'inferFunc',@UGM_Infer_TRBP ...
				 ,'Cvec',.01 ...%[.01 .1 1] ...
				 ,'Cvec2',.001 ...%[.01 .05 .1 .5 1] ...
				 ,'stepSizeVec',.02 ...
				 ,'kappaVec',1 ... %[.1 .2 .5 1 2] ...
				 ,'computeBaseline',examples{1}.nNodeFeat==2 ...
				 );
expSetup.optSGD = struct('maxIter',100 ...
						,'plotObj',103,'plotRefresh',10 ...
						,'verbose',1,'returnBest',1 ...
						);
expSetup.optLBFGS = struct('MaxIter',100,'MaxFunEvals',100 ...
						  ,'plotObj',103,'plotRefresh',10 ...
						  ,'Display','off','verbose',3 ...
						  );
expSetup.plotPred = 102;

experiment


%% Plot Noisy NIPS

clear
load results/nips_n15_trbp.mat

f = 1;
teidx = foldIdx(f).teidx;
ex_te = examples(teidx);
vctsmIdx = 1;
m3nIdx = 3;

err = zeros(3,10);
for i = 1:10
	
	ex = ex_te{i};
	baseline = squeeze(ex.Xnode(1,2,:)==1) + 1;
	err(1,i) = nnz(ex.Y ~= baseline) / ex.nNode;
	fprintf('Baseline error:\t%f\n', err(1,i));
	figure(1)
	imagesc(reshape(ex.Xnode(1,2,:),42,60)); colormap gray; axis off; set(gca,'Position',[0 0 1 1]);

	w = params{m3nIdx,f,bestParam(m3nIdx,f)}.w;
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
	pred = decodeFunc(nodePot,edgePot,ex.edgeStruct);
	err(2,i) = nnz(ex.Y ~= pred) / ex.nNode;
	fprintf('M3N error:\t%f\n', err(2,i));
	figure(2)
	imagesc(reshape(pred,42,60)); colormap gray; axis off; set(gca,'Position',[0 0 1 1]);

	w = params{vctsmIdx,f,bestParam(vctsmIdx,f)}.w;
	kappa = params{vctsmIdx,f,bestParam(vctsmIdx,f)}.kappa;
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
	pred = UGM_Decode_ConvexBP(kappa,nodePot,edgePot,ex.edgeStruct,inferFunc);
	err(3,i) = nnz(ex.Y ~= pred) / ex.nNode;
	fprintf('VCTSM error:\t%f\n', err(3,i));
	figure(3)
	imagesc(reshape(pred,42,60)); colormap gray; axis off; set(gca,'Position',[0 0 1 1]);

	pause
end
mean(err,2)

