% Experimental testing harness
%
% Requires 1 variable, 1 optional:
% 	examples : cell array of (labeled) examples
% 	expSetup : (optional) structure containing experimental setup

assert(exist('examples','var') && iscell(examples), 'experiment requires cell array of examples.');
nEx = length(examples);
nParam = max(examples{1}.edgeMap(:));

% experiment vars
if ~exist('expSetup','var')
	expSetup = struct();
end
if isfield(expSetup,'nFold')
	nFold = expSetup.nFold;
else
	nFold = 1;
end
if isfield(expSetup,'foldDist')
	foldDist = expSetup.foldDist;
	nExFold = sum(foldDist);
	assert(nExFold <= nEx, 'Number of examples per fold greater than examples.');
	nTrain = foldDist(1);
	nUnlab = foldDist(2);
	nCV = foldDist(3);
	nTest = foldDist(4);
else
	nExFold = nEx / nFold;
	nTrain = 1;
	nUnlab = 1;
	nCV = 1;
	nTest = nExFold - 3;
end
algoNames = {'MLE','M3N','M3NLRR','VCTSM','CACC','CSM3N','CSCACC','DLM'};
if isfield(expSetup,'runAlgos')
	runAlgos = expSetup.runAlgos;
else
	runAlgos = 1:length(algoNames);
end
if isfield(expSetup,'Cvec')
	Cvec = expSetup.Cvec;
else
	Cvec = [0; reshape([.5*10.^linspace(0,4,5);10.^linspace(0,4,5)],[],1)];
end
if isfield(expSetup,'decoder')
	decoder = expSetup.decoder;
else
	decoder = @UGM_Decode_LBP;
end
if isfield(expSetup,'edgeFeatFunc')
	edgeFeatFunc = expSetup.edgeFeatFunc;
else
	edgeFeatFunc = @UGM_makeEdgeFeatures;
end
if isfield(expSetup,'discreteX')
	discreteX = expSetup.discreteX;
else
	discreteX = 1;
end
if isfield(expSetup,'nStabSamp')
	nStabSamp = expSetup.nStabSamp;
else
	nStabSamp = 10;
end

nRunAlgos = length(runAlgos);
nCvals = length(Cvec);


%% MAIN LOOP

% job metadata
nJobs = nRunAlgos * nCvals * nFold;
totalTimer = tic;
count = 0;

% storage
params = cell(nRunAlgos,nCvals,nFold);
trErrs = zeros(nRunAlgos,nCvals,nFold);
cvErrs = zeros(nRunAlgos,nCvals,nFold);
teErrs = zeros(nRunAlgos,nCvals,nFold);
cvStab = zeros(nRunAlgos,nCvals,nFold,2);

% best parameters based on {CV,stab,test)
bestParamCV = zeros(nRunAlgos,nFold);
bestParamStab = zeros(nRunAlgos,nFold);
bestParamTest = zeros(nRunAlgos,nFold);

for fold = 1:nFold
	
	if (fold * nExFold) > nEx
		break;
	end
	
	fprintf('Starting fold %d of %d.\n', fold,nFold);
	
	% separate training/CV/testing
	fidx = (fold-1) * nExFold;
	tridx = fidx+1:fidx+nTrain;
	ulidx = fidx+nTrain+1:fidx+nTrain+nUnlab;
	cvidx = fidx+nTrain+nUnlab+1:fidx+nTrain+nUnlab+nCV;
	teidx = fidx+nTrain+nUnlab+nCV+1:fidx+nTrain+nUnlab+nCV++nTest;
	ex_tr = examples(tridx);
	ex_ul = examples(ulidx);
	ex_cv = examples(cvidx);
	ex_te = examples(teidx);
	
	% init perturbations for stability measurement
	perturbs = [];
	
	for c = 1:nCvals
		C = Cvec(c);
		
		for a = 1:nRunAlgos

			%% TRAINING

			switch(runAlgos(a))
				
				% M(P)LE learning
				case 1
					fprintf('Training MLE ...\n');
					[w,nll] = trainMLE(ex_tr,@UGM_Infer_MeanField,C);
					params{a,c,fold}.w = w;

				% M3N learning
				case 2
					fprintf('Training M3N ...\n');
					[w,fAvg] = trainM3N(ex_tr,decoder,C);
					params{a,c,fold}.w = w;

				% M3NLRR learning (M3N with separate local/relational reg.)
				case 3
					fprintf('Training M3N with local/relational regularization ...\n');
					maxLocParamIdx = max(ex_tr{1}.nodeMap(:));
					relMultiplier = 10; % hack
					Csplit = C * ones(nParam,1);
					Csplit(maxLocParamIdx+1:end) = Csplit(maxLocParamIdx+1:end) * relMultiplier;
					[w,fAvg] = trainM3N(ex_tr,decoder,Csplit);
					params{a,c,fold}.w = w;

				% VCTSM learning (convexity optimization)
				case 4
					fprintf('Training VCTSM ...\n');
					[w,kappa,f] = trainVCTSM(ex_tr,C);
					params{a,c,fold}.w = w;
					params{a,c,fold}.kappa = kappa;

				% CACC learning (robust M3N)
				case 5
					fprintf('Training CACC ...\n');
					[w,fAvg] = trainCACC(ex_tr,decoder,C);
					params{a,c,fold}.w = w;
					
				% CSM3N learning (M3N + stability reg.)
				case 6
					fprintf('Training CSM3N ...\n');
					[w,fAvg] = trainCSM3N(ex_tr,ex_ul,decoder,C,.25);
					params{a,c,fold}.w = w;
					
				% CSCACC learning (CACC + stability reg.)
				case 7
					fprintf('Training CSCACC ...\n');
					[w,fAvg] = trainCSCACC(ex_tr,ex_ul,decoder,C,.25);
					params{a,c,fold}.w = w;
				
				% DLM learning
				case 8
					fprintf('Training DLM ...\n');
					[w,fAvg] = trainDLM(ex_tr,decoder,C);
					params{a,c,fold}.w = w;
					
			end
			
			% training stats
			errs = zeros(nTrain,1);
			for i = 1:nTrain
				ex = ex_tr{i};
				if a ~= 4
					[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
					pred = decoder(nodePot,edgePot,ex.edgeStruct);
				else
					mu = vctsmInfer(w,kappa,ex.Fx,ex.Aeq,ex.beq);
					pred = decodeMarginals(mu,ex.nNode,ex.nState);
				end
				errs(i) = nnz(ex.Y ~= pred) / ex.nNode;
			end
			trErrs(a,c,fold) = sum(errs)/nTrain;
			fprintf('Avg train err = %.4f\n', trErrs(a,c,fold));
			
			%% CROSS-VALIDATION
			
			errs = zeros(nCV,1);
			stab = zeros(nCV,2);
			for i = 1:nCV
				ex = ex_cv{i};
				if a ~= 4
					[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
					pred = decoder(nodePot,edgePot,ex.edgeStruct);
					if nStabSamp > 0
						[stab(i,1),stab(i,2),perturbs] = measureStabilityRand({w},ex,discreteX,nStabSamp,decoder,edgeFeatFunc,pred,perturbs);
					end
				else
					mu = vctsmInfer(w,kappa,ex.Fx,ex.Aeq,ex.beq);
					pred = decodeMarginals(mu,ex.nNode,ex.nState);
					if nStabSamp > 0
						[stab(i,1),stab(i,2),perturbs] = measureStabilityRand({w,kappa},ex,discreteX,nStabSamp,[],edgeFeatFunc,pred,perturbs);
					end
				end
				errs(i) = nnz(ex.Y ~= pred()) / ex.nNode;
			end
			cvErrs(a,c,fold) = mean(errs);
			fprintf('Avg CV err = %.4f\n', cvErrs(a,c,fold));
			if nStabSamp > 0
				cvStab(a,c,fold,:) = [max(stab(:,1)) mean(stab(:,2))];
				fprintf('CV stab: max = %d, avg = %.4f\n', cvStab(a,c,fold,1),cvStab(a,c,fold,2));
			else
				cvStab(a,c,fold,:) = [0 0];
			end

			%% TESTING
			
			errs = zeros(nTest,1);
			for i = 1:nTest
				ex = ex_te{i};
				if a ~= 4
					[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
					pred = decoder(nodePot,edgePot,ex.edgeStruct);
				else
					mu = vctsmInfer(w,kappa,ex.Fx,ex.Aeq,ex.beq);
					pred = decodeMarginals(mu,ex.nNode,ex.nState);
				end
				errs(i) = nnz(ex.Y ~= pred()) / ex.nNode;
				% plot prediction
				%subplot(nRunAlgos,1,a);
				%imagesc(reshape(pred,32,32));
				%title(algoNames(a));
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
	
	% choose best parameters
	for a = 1:nRunAlgos
		bestParamCV(a,fold) = find(cvErrs(a,:,fold)==min(cvErrs(a,:,fold)),1,'first');
		bestParamStab(a,fold) = find(cvStab(a,:,fold,1)==min(cvStab(a,:,fold,1)),1,'first');
		bestParamTest(a,fold) = find(teErrs(a,:,fold)==min(teErrs(a,:,fold)),1,'first');
	end
	
	fprintf('\n');

end

% generalization error
geErrs = teErrs - trErrs;

% display results at end
colStr = {'Train','Valid','Test','GenErr','MaxStab','AvgStab','C'};
bestParam = bestParamCV;
bestResults = zeros(nRunAlgos,length(colStr),nFold);
for fold = 1:nFold
	idx = sub2ind([nRunAlgos nCvals nFold],(1:nRunAlgos)',bestParam(:,fold),fold*ones(nRunAlgos,1));
	bestResults(:,:,fold) = [trErrs(idx) cvErrs(idx) teErrs(idx) geErrs(idx) cvStab(idx) cvStab(idx+numel(teErrs)) Cvec(bestParam(:,fold))];
	disptable(bestResults(:,:,fold),colStr,algoNames(runAlgos),'%.5f');
end

% compute mean/stdev across folds
avgResults = mean(bestResults,3);
stdResults = std(bestResults,[],3);



