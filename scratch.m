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


%% Train CSM3N
clear;
examples = noisyX(10,1,0,1,0);
ex_l = examples(1:5);
ex_u = examples(6:10);
[w,fAvg] = trainCSM3N(ex_l,ex_u,@UGM_Decode_LBP,100)


%% Run noisyX experiment
clear;
examples = noisyX(4,2,0,0,0);
Xdesc = struct('discreteX',0,'nonneg',0);
expSetup = struct('nFold',1,'foldDist',[1 1 1 1],'runAlgos',2,...
				  'Cvec',[0 1 5 10 50 100 500 1000 5000 10000],'nStabSamp',10,'Xdesc',Xdesc,...
				  'decodeFunc',@UGM_Decode_LBP,'inferFunc',@UGM_Infer_LBP);
experiment;


%% Run PoliBlog experiment
clear;
cd data/poliblog/Processed;
[examples,foldIdx] = loadPoliBlog();
cd ../../..;
Xdesc = struct('discreteX',1);
expSetup = struct('foldIdx',foldIdx,'Xdesc',Xdesc,...
				  'runAlgos',[1 2 4],...
				  'decodeFunc',@UGM_Decode_LBP,'inferFunc',@UGM_Infer_LBP,...
				  'nStabSamp',0,...
				  'Cvec',[.1 .5 1 5 10 50 100 500 1000 5000 10000],'CvecRel',0,...
				  'save2file','poliblog1.mat');
expSetup.optSGD = struct('maxIter',500);
experiment;


%% Run Cora experiment
clear;
cd data;
[examples,foldIdx] = loadDocData('cora/cora.mat',4,20,0);
cd ..;
Xdesc = struct('discreteX',0,'nonneg',0);
expSetup = struct('foldIdx',foldIdx,'Xdesc',Xdesc,...
				  'runAlgos',[1 2 4],...
				  'decodeFunc',@UGM_Decode_LBP,'inferFunc',@UGM_Infer_LBP,...
				  'nStabSamp',0,...
				  'Cvec',[.1 .5 1 5 10 50 100 500 1000 5000 10000],'CvecRel',0,...
				  'save2file','cora1.mat');
expSetup.optSGD = struct('maxIter',200);
experiment;


