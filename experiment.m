% Experimental testing harness

% experiment vars
nEx = length(examples);
nFold = 10;
nExFold = nEx / nFold;
nTrain = nExFold - 2;
nCV = 1;
nTest = 1;

% algorithm vars
algoNames = {'MLE', 'M3N', 'CSM3N'};
algoTypes = 2:3;%1:5;

% model parameters
nParam = max(max(examples{1}.nodeMap(:)),max(examples{1}.edgeMap(:)));

% crossvalidation vars
Cvec = 1024;%10.^linspace(-2,6,9);

% stability vars
% maxSamp = 10;
% nStabSamp = min(maxSamp, nNode*(nState-1));


%% MAIN LOOP

% job metadata
nJobs = length(algoTypes) * length(Cvec) * nFold;
totalTimer = tic;
count = 0;

% storage
params = cell(length(algoTypes), length(Cvec), nFold);
trErrs = zeros(length(algoTypes), length(Cvec), nFold);
cvErrs = zeros(length(algoTypes), length(Cvec), nFold);
teErrs = zeros(length(algoTypes), length(Cvec), nFold);

for fold = 1:nFold
	
	% separate training/CV/testing
	fidx = (fold-1) * nExFold;
	tridx = fidx+1:fidx+nTrain;
	cvidx = fidx+nTrain+1:fidx+nTrain+nCV;
	teidx = fidx+nTrain+nCV+1:fidx+nExFold;
	ex_tr = examples(tridx);
	ex_cv = examples(cvidx);
	ex_te = examples(teidx);
	
	
	for c = 1:length(Cvec)
		C = Cvec(c);
		
		for a = 1:length(algoTypes)

			%% TRAINING

			% none of the training methods seem to work with mex
			for i = 1:nTrain
				ex_tr{i}.edgeStruct.useMex = 0;
			end
			switch(algoTypes(a))
				
				% random weights (for bypassing learning)
				case 0
					w = rand(nParam,1);

				% M(P)LE learning
				case 1
					%[w,nll] = trainMLE(ex_tr,@UGM_Infer_MeanField,C)
					[w,nll] = trainMLE(ex_tr,0,C)

				% M3N learning
				case 2
					m3nDecoder = @(nodePot,edgePot,edgeStruct) UGM_Decode_MaxOfMarginals(nodePot,edgePot,edgeStruct,@UGM_Infer_LBP);
					[w,loss] = trainM3N(ex_tr,m3nDecoder,C,2)

				% CSM3N1 learning (M3N with separate local/relational regularization)
				case 3
					maxLocParamIdx = max(ex_tr{1}.nodeMap(:));
					relMultiplier = 10; % hack
					Csplit = C * ones(nParam,1);
					Csplit(1:maxLocParamIdx) = Csplit(1:maxLocParamIdx) * 0.5*relMultiplier;
					m3nDecoder = @(nodePot,edgePot,edgeStruct) UGM_Decode_MaxOfMarginals(nodePot,edgePot,edgeStruct,@UGM_Infer_LBP);
					[w,loss] = trainM3N(ex_tr,m3nDecoder,Csplit,2)

				% CSM3N2 learning (convexity optimization)
				case 4

				% CSM3N3 learning (stability regularization)
				case 5

			end
			% turn mex back on
			for i = 1:nTrain
				ex_tr{i}.edgeStruct.useMex = 1;
			end
			
			% training stats
			errs = zeros(nTrain,1);
			for i = 1:nTrain
				[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex_tr{i}.Xnode,ex_tr{i}.Xedge,ex_tr{i}.nodeMap,ex_tr{i}.edgeMap,ex_tr{i}.edgeStruct);
				%pred = UGM_Decode_LinProg(nodePot,edgePot,edgeStruct);
				pred = UGM_Decode_MaxOfMarginals(nodePot,edgePot,ex_tr{i}.edgeStruct,@UGM_Infer_LBP);
				errs(i) = nnz(ex_tr{i}.Y ~= pred(1:ex_tr{i}.nNode)) / ex_tr{i}.nNode;
			end
			trErrs(a,c,fold) = sum(errs)/nTrain;
			fprintf('Avg train err = %.4f\n', trErrs(a,c,fold));
			
			%% CROSS-VALIDATION
			
			errs = zeros(nCV,1);
			for i = 1:nCV
				[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex_cv{i}.Xnode,ex_cv{i}.Xedge,ex_cv{i}.nodeMap,ex_cv{i}.edgeMap,ex_cv{i}.edgeStruct);
				%pred = UGM_Decode_LinProg(nodePot,edgePot,edgeStruct);
				pred = UGM_Decode_MaxOfMarginals(nodePot,edgePot,ex_cv{i}.edgeStruct,@UGM_Infer_LBP);
				errs(i) = nnz(ex_cv{i}.Y ~= pred(1:ex_cv{i}.nNode)) / ex_cv{i}.nNode;
			end
			cvErrs(a,c,fold) = sum(errs)/nCV;
			fprintf('Avg CV err = %.4f\n', cvErrs(a,c,fold));
			
			%% TESTING
			
			errs = zeros(nTest,1);
			for i = 1:nTest
				[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex_te{i}.Xnode,ex_te{i}.Xedge,ex_te{i}.nodeMap,ex_te{i}.edgeMap,ex_te{i}.edgeStruct);
				%pred = UGM_Decode_LinProg(nodePot,edgePot,edgeStruct);
				pred = UGM_Decode_MaxOfMarginals(nodePot,edgePot,ex_te{i}.edgeStruct,@UGM_Infer_LBP);
				errs(i) = nnz(ex_te{i}.Y ~= pred(1:ex_te{i}.nNode)) / ex_te{i}.nNode;
			end
			teErrs(a,c,fold) = sum(errs)/nTest;
			fprintf('Avg test err = %.4f\n', teErrs(a,c,fold));
			
			%% PROGRESS
			
			count = count + 1;
			curTime = toc(totalTimer);
			fprintf('Finished %d of %d; elapsed: %.2f min; ETA: %.2f min\n', ...
				count, nJobs, curTime/60, (nJobs-count)*(curTime/count)/60);
			
		end

	end

end
