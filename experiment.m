% Experimental testing harness

% experiment vars
nEx = length(examples);
nFold = 1;
nExFold = nEx / nFold;
nTrain = 3;%nExFold - 2;
nCV = 1;
nTest = 1;

% algorithm vars
algoNames = {'MLE', 'M3N', 'CSM3N'};
algoTypes = [1 2 4];%1:5;

% model parameters
nParam = max(examples{1}.edgeMap(:));

% crossvalidation vars
Cvec = 100;%10.^linspace(-2,6,9);

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
	teidx = fidx+nTrain+nCV+1:fidx+nTest;
	ex_tr = examples(tridx);
	ex_cv = examples(cvidx);
	ex_te = examples(teidx);
	
	
	for c = 1:length(Cvec)
		C = Cvec(c);
		
		for a = 1:length(algoTypes)

			%% TRAINING

			switch(algoTypes(a))
				
				% M(P)LE learning
				case 1
					fprintf('Training MLE ...\n');
					[w,nll] = trainMLE(ex_tr,@UGM_Infer_MeanField,C)
					%[w,nll] = trainMLE(ex_tr,0,C)

				% M3N learning
				case 2
					fprintf('Training M3N ...\n');
					m3nDecoder = @(nodePot,edgePot,edgeStruct) UGM_Decode_MaxOfMarginals(nodePot,edgePot,edgeStruct,@UGM_Infer_LBP);
					[w,loss] = trainM3N(ex_tr,m3nDecoder,C)

				% CSM3N1 learning (M3N with separate local/relational regularization)
				case 3
					fprintf('Training M3N with local/relational regularization ...\n');
					maxLocParamIdx = max(ex_tr{1}.nodeMap(:));
					relMultiplier = 10; % hack
					Csplit = C * ones(nParam,1);
					Csplit(1:maxLocParamIdx) = Csplit(1:maxLocParamIdx) * 0.5*relMultiplier;
					m3nDecoder = @(nodePot,edgePot,edgeStruct) UGM_Decode_MaxOfMarginals(nodePot,edgePot,edgeStruct,@UGM_Infer_LBP);
					[w,loss] = trainM3N(ex_tr,m3nDecoder,Csplit)

				% CSM3N2 learning (convexity optimization)
				case 4
					fprintf('Training CSM3N ...\n');
					[w,kappa,f] = trainVCTSM(ex_tr,C)

				% CSM3N3 learning (stability regularization)
				case 5

			end
			continue;
			
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
