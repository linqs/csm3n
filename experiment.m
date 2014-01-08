% Experimental testing harness

% experiment vars
nEx = length(examples);
nFold = 1;
nExFold = nEx / nFold;
nTrain = nExFold - 2;
nCV = 1;
nTest = 1;

% algorithm vars
algoNames = {'MLE', 'M3N', 'M3NLRR', 'VCTSM'};
runAlgos = 1:4;

% model parameters
nParam = max(examples{1}.edgeMap(:));

% crossvalidation vars
Cvec = 100;%10.^linspace(-2,6,9);

% stability vars
% maxSamp = 10;
% nStabSamp = min(maxSamp, nNode*(nState-1));


%% MAIN LOOP

% job metadata
nJobs = length(runAlgos) * length(Cvec) * nFold;
totalTimer = tic;
count = 0;

% storage
params = cell(length(runAlgos), length(Cvec), nFold);
trErrs = zeros(length(runAlgos), length(Cvec), nFold);
cvErrs = zeros(length(runAlgos), length(Cvec), nFold);
teErrs = zeros(length(runAlgos), length(Cvec), nFold);

for fold = 1:nFold
	
	% separate training/CV/testing
	fidx = (fold-1) * nExFold;
	tridx = fidx+1:fidx+nTrain;
	cvidx = fidx+nTrain+1:fidx+nTrain+nCV;
	teidx = fidx+nTrain+nCV+1:fidx+nTrain+nCV++nTest;
	ex_tr = examples(tridx);
	ex_cv = examples(cvidx);
	ex_te = examples(teidx);
	
	
	for c = 1:length(Cvec)
		C = Cvec(c);
		
		for a = 1:length(runAlgos)

			%% TRAINING

			switch(runAlgos(a))
				
				% M(P)LE learning
				case 1
					fprintf('Training MLE ...\n');
					[w,nll] = trainMLE(ex_tr,@UGM_Infer_MeanField,C)
					params{a,c,fold}.w = w;

				% M3N learning
				case 2
					fprintf('Training M3N ...\n');
					[w,fAvg] = trainM3N(ex_tr,@UGM_Decode_LBP,C)
					params{a,c,fold}.w = w;

				% M3NLRR learning (M3N with separate local/relational regularization)
				case 3
					fprintf('Training M3N with local/relational regularization ...\n');
					maxLocParamIdx = max(ex_tr{1}.nodeMap(:));
					relMultiplier = 100; % hack
					Csplit = C * ones(nParam,1);
					Csplit(maxLocParamIdx+1:end) = Csplit(maxLocParamIdx+1:end) * relMultiplier;
					[w,fAvg] = trainM3N(ex_tr,@UGM_Decode_LBP,Csplit)
					params{a,c,fold}.w = w;

				% VCTSM learning (convexity optimization)
				case 4
					fprintf('Training VCTSM ...\n');
					[w,kappa,f] = trainVCTSM(ex_tr,C)
					params{a,c,fold}.w = w;
					params{a,c,fold}.kappa = kappa;

				% CSM3N learning (stability regularization)
				case 5

			end
			
			% training stats
			errs = zeros(nTrain,1);
			for i = 1:nTrain
				if ismember(runAlgos(a),1:3)
					[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex_tr{i}.Xnode,ex_tr{i}.Xedge,ex_tr{i}.nodeMap,ex_tr{i}.edgeMap,ex_tr{i}.edgeStruct);
					pred = UGM_Decode_LBP(nodePot,edgePot,ex_tr{i}.edgeStruct);
				else
					mu = vctsmInfer(w,kappa,ex_tr{i}.F,ex_tr{i}.Aeq,ex_tr{i}.beq);
					pred = decodeMarginals(mu, ex_tr{i}.nNode, ex_tr{i}.nState);
				end
				errs(i) = nnz(ex_tr{i}.Y ~= pred(1:ex_tr{i}.nNode)) / ex_tr{i}.nNode;
			end
			trErrs(a,c,fold) = sum(errs)/nTrain;
			fprintf('Avg train err = %.4f\n', trErrs(a,c,fold));
			
			%% CROSS-VALIDATION
			
			errs = zeros(nCV,1);
			for i = 1:nCV
				if ismember(runAlgos(a),1:3)
					[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex_cv{i}.Xnode,ex_cv{i}.Xedge,ex_cv{i}.nodeMap,ex_cv{i}.edgeMap,ex_cv{i}.edgeStruct);
					pred = UGM_Decode_LBP(nodePot,edgePot,ex_cv{i}.edgeStruct);
				else
					mu = vctsmInfer(w,kappa,ex_cv{i}.F,ex_cv{i}.Aeq,ex_cv{i}.beq);
					pred = decodeMarginals(mu, ex_cv{i}.nNode, ex_cv{i}.nState);
				end
				errs(i) = nnz(ex_cv{i}.Y ~= pred(1:ex_cv{i}.nNode)) / ex_cv{i}.nNode;
			end
			cvErrs(a,c,fold) = sum(errs)/nCV;
			fprintf('Avg CV err = %.4f\n', cvErrs(a,c,fold));
			
			%% TESTING
			
			errs = zeros(nTest,1);
			for i = 1:nTest
				if ismember(runAlgos(a),1:3)
					[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex_te{i}.Xnode,ex_te{i}.Xedge,ex_te{i}.nodeMap,ex_te{i}.edgeMap,ex_te{i}.edgeStruct);
					pred = UGM_Decode_LBP(nodePot,edgePot,ex_te{i}.edgeStruct);
				else
					mu = vctsmInfer(w,kappa,ex_te{i}.F,ex_te{i}.Aeq,ex_te{i}.beq);
					pred = decodeMarginals(mu, ex_te{i}.nNode, ex_te{i}.nState);
				end
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

% generalization error
geErrs = teErrs - trErrs;

% display results
[trErrs cvErrs teErrs geErrs]

