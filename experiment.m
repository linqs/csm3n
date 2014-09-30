% Experimental testing harness
%
% Requires 1 variable, 1 optional:
% 	examples : cell array of (labeled) examples
% 	expSetup : (optional) structure containing experimental setup

clearvars -except examples expSetup;
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
if isfield(expSetup,'foldIdx')
	foldIdx = expSetup.foldIdx;
	nFold = length(foldIdx);
elseif isfield(expSetup,'foldDist')
	foldDist = expSetup.foldDist;
	assert(sum(foldDist) <= nEx, 'Number of examples per fold greater than examples.');
	foldIdx = makeFolds(nEx,nFold,foldDist);
	nFold = min(nFold,length(foldIdx));
else
	nExFold = floor(nEx / nFold);
	nTrain = 1;
	nUnlab = 1;
	nCV = 1;
	nTest = nExFold - 3;
	foldIdx = makeFolds(nEx,nFold,nTrain,nUnlab,nCV,nTest);
	nFold = min(nFold,length(foldIdx));
end

algoNames = {'MLE','PERC','M3N','M3NFW','SCTSM','VCTSM','VCTSMlog'};
if isfield(expSetup,'runAlgos')
	runAlgos = expSetup.runAlgos;
else
	runAlgos = 1:length(algoNames);
end
nRunAlgos = length(runAlgos);

% Optimization options
if isfield(expSetup,'optSGD')
	optSGD = expSetup.optSGD;
else
	optSGD = struct();
end
if isfield(expSetup,'optLBFGS')
	optLBFGS = expSetup.optLBFGS;
else
	optLBFGS = struct();
end
% Algorithm-specific optimization options
if isfield(expSetup,'optMLE')
	optMLE = expSetup.optMLE;
else
	optMLE = optLBFGS;
end
if isfield(expSetup,'optPerc')
	optPerc = expSetup.optPerc;
else
	optPerc = optSGD;
end
if isfield(expSetup,'optM3N')
	optM3N = expSetup.optM3N;
else
	optM3N = optSGD;
end
if isfield(expSetup,'optM3NFW')
	optM3NFW = expSetup.optM3NFW;
else
	optM3NFW = optSGD;
end
if isfield(expSetup,'optSCTSM')
	optSCTSM = expSetup.optSCTSM;
else
	optSCTSM = optLBFGS;
end
if isfield(expSetup,'optVCTSM')
	optVCTSM = expSetup.optVCTSM;
else
	optVCTSM = optLBFGS;
end

% Hyperparameters
if isfield(expSetup,'Cvec')
	Cvec = expSetup.Cvec;
else
	Cvec = 1;
end
if isfield(expSetup,'kappaVec')
	kappaVec = expSetup.kappaVec;
elseif any(runAlgos==5)
	kappaVec = [.1 .2 .5 1 2 5 10];
else
	kappaVec = 1;
end
if isfield(expSetup,'stepSizeVec')
	stepSizeVec = expSetup.stepSizeVec;
elseif isfield(optM3N,'stepSize')
	stepSizeVec = optM3N.stepSize;
else
	stepSizeVec = 1;
end
nCvals1 = length(Cvec);
nCvals2 = max([length(kappaVec),length(stepSizeVec)]);

% Init kappa for VCTSM
if isfield(expSetup,'initKappa')
	initKappa = expSetup.initKappa;
else
	initKappa = 1;
end

% Inference/feature algos
if isfield(expSetup,'decodeFunc')
	decodeFunc = expSetup.decodeFunc;
else
	decodeFunc = @UGM_Decode_TRBP;
end
if isfield(expSetup,'inferFunc')
	inferFunc = expSetup.inferFunc;
else
	inferFunc = @UGM_Infer_TRBP;
end
if isfield(expSetup,'edgeFeatFunc')
	edgeFeatFunc = expSetup.edgeFeatFunc;
else
	edgeFeatFunc = @makeEdgeFeatures;
end

% File to save workspace
if isfield(expSetup,'save2file')
	save2file = expSetup.save2file;
else
	save2file = [];
end

% Plot last predictions
if isfield(expSetup,'plotPred')
	plotPred = expSetup.plotPred;
else
	plotPred = 0;
end
if isfield(expSetup,'plotFunc')
	plotFunc = expSetup.plotFunc;
else
	plotFunc = [];
end

% Compute baselines?
if isfield(expSetup,'computeBaseline')
	computeBaseline = expSetup.computeBaseline;
else
	computeBaseline = 0;
end

% Use full training for stats?
if isfield(expSetup,'useFullTrain')
	useFullTrain = expSetup.useFullTrain;
else
	useFullTrain = 0;
end


%% MAIN LOOP

% Job metadata
nJobs = nFold * nCvals1 * (...
	length(intersect(runAlgos,[1 4 6 7])) + ...
	length(stepSizeVec) * length(intersect(runAlgos,[2 3])) + ...
	length(kappaVec) * any(runAlgos==5));
totalTimer = tic;
count = 0;

% Parameter storage
params = cell(nRunAlgos,nFold,nCvals1,nCvals2);
% Stats storage
trErrs = inf(nRunAlgos,nFold,nCvals1,nCvals2);
cvErrs = inf(nRunAlgos,nFold,nCvals1,nCvals2);
teErrs = inf(nRunAlgos,nFold,nCvals1,nCvals2);
teF1 = inf(nRunAlgos,nFold,nCvals1,nCvals2);
% Baseline stats
baselineErrs = inf(nFold,1);
baselineF1 = inf(nFold,1);

% Best parameters based on {CV,stab,test)
bestParamTrain = zeros(nRunAlgos,nFold);
bestParamCVerr = zeros(nRunAlgos,nFold);
bestParamStab = zeros(nRunAlgos,nFold);
bestParamTest = zeros(nRunAlgos,nFold);

% Stats for training on [tr cv]
paramsFull = cell(nRunAlgos,nFold);
trErrsFull = inf(nRunAlgos,nFold);
teErrsFull = inf(nRunAlgos,nFold);
teF1Full = inf(nRunAlgos,nFold);

% Number of local parameters
nLocParam = max(examples{1}.nodeMap(:));


for fold = 1:nFold
	
	fprintf('Starting fold %d of %d.\n', fold,nFold);
	
	% Separate training/CV/testing
	tridx = foldIdx(fold).tridx;
	ulidx = foldIdx(fold).ulidx;
	cvidx = foldIdx(fold).cvidx;
	teidx = foldIdx(fold).teidx;
	nTrain = length(tridx);
	nUnlab = length(ulidx);
	nCV = length(cvidx);
	nTest = length(teidx);
	ex_tr = examples(tridx);
	ex_ul = examples(ulidx);
	ex_cv = examples(cvidx);
	ex_te = examples(teidx);
	
	% Init perturbations for stability measurement
	pert = [];
	
	% Compute baseline stats
	if computeBaseline
		errs = zeros(nTest,1);
		f1 = zeros(nTest,1);
		for i = 1:nTest
			ex = ex_te{i};
			pred = zeros(ex.nNode,1);
			for n = 1:ex.nNode
				pred(n) = find(ex.Xnode(1,:,n));
			end
			errs(i) = nnz(ex.Y ~= pred) / ex.nNode;
			[~,~,~,~,s.f1,~,s.f1wavg] = errStats(double(ex.Y),pred);
			if ex.nState == 2
				f1(i) = s.f1(1); % If binary, use F1 of first class
			else
				f1(i) = s.f1wavg; % If multiclass, use weighted average F1
			end
		end
		baselineErrs(fold) = mean(errs);
		baselineF1(fold) = mean(f1);
		fprintf('Baseline: avg err = %.4f, avg F1 = %.4f\n', baselineErrs(fold),baselineF1(fold));
	end
	
	for c1 = 1:nCvals1
		C_w = Cvec(c1);
		
		for c2 = 1:nCvals2
			
			for a = 1:nRunAlgos
				
				% Select hyperparameters for training
				if strcmp(algoNames{runAlgos(a)},'M3N') || strcmp(algoNames{runAlgos(a)},'PERC')
					% M3N
					if c2 > length(stepSizeVec)
						continue
					end
					stepSize = stepSizeVec(c2);
				elseif strcmp(algoNames{runAlgos(a)},'SCTSM')
					% SCTSM
					if c2 > length(kappaVec)
						continue
					end
					kappa = kappaVec(c2);
				elseif c2 > 1
					% Some algorithms don't use second reg. param.
					continue
				end
				
				%% TRAINING
				
				try % Might get an exception during training for certain algos
				
				switch(runAlgos(a))
					
					% M(P)LE learning
					case 1
						fprintf('Training MLE with C=%f \n',C_w);
						[w,nll] = trainMLE(ex_tr,inferFunc,C_w,optVCTSM);
						params{a,fold,c1,c2}.w = w;
						
					% Perceptron learning
					case 2
						fprintf('Training Perceptron with C=%f, stepSize=%f \n',C_w,stepSize);
						optM3N.stepSize = stepSize;
						[w,fAvg] = trainPerceptron(ex_tr,decodeFunc,C_w,optM3N);
						params{a,fold,c1,c2}.w = w;
						
					% M3N learning
					case 3
						fprintf('Training M3N with C=%f, stepSize=%f \n',C_w,stepSize);
						optM3N.stepSize = stepSize;
						[w,fAvg] = trainM3N(ex_tr,decodeFunc,C_w,optM3N);
						params{a,fold,c1,c2}.w = w;
						
					% M3N learning with FW
					case 4
						fprintf('Training M3NFW with C=%f ...\n',C_w);
						[w,fAvg] = bcfw(ex_tr,decodeFunc,C_w,optM3N);
						params{a,fold,c1,c2}.w = w;
												
					% SCTSM learning (fixed convexity)
					case 5
						fprintf('Training SCTSM with C=%f, kappa=%f \n',C_w,kappa);
						[w,f] = trainSCTSM(ex_tr,inferFunc,kappa,C_w,optSCTSM);
						params{a,fold,c1,c2}.w = w;
						
					% VCTSM learning (convexity optimization)
					case 6
						fprintf('Training VCTSM with C=%f ...\n',C_w);
						[w,kappa,f] = trainVCTSM(ex_tr,inferFunc,C_w,optVCTSM,[],initKappa);
						params{a,fold,c1,c2}.w = w;
						params{a,fold,c1,c2}.kappa = kappa;
						
					% VCTSM_log learning (log version)
					case 7
						fprintf('Training VCTSMlog with C=%f ...\n',C_w);
						[w,kappa,f] = trainVCTSM_log(ex_tr,inferFunc,C_w,optVCTSM,[],initKappa);
						params{a,fold,c1,c2}.w = w;
						params{a,fold,c1,c2}.kappa = kappa;
						
				end
				
				catch exception % Caught an exception during training
					fprintf('Caught exception : %s\n  Skipping current job.', exception.message);
					continue
				end
				
				% Training stats
				errs = zeros(nTrain,1);
				for i = 1:nTrain
					ex = ex_tr{i};
					[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
					if strcmp(algoNames{runAlgos(a)},'SCTSM') ...
					|| strcmp(algoNames{runAlgos(a)},'VCTSM') ...
					|| strcmp(algoNames{runAlgos(a)},'VCTSMlog')
						pred = UGM_Decode_ConvexBP(kappa,nodePot,edgePot,ex.edgeStruct,inferFunc);
					else
						pred = decodeFunc(nodePot,edgePot,ex.edgeStruct);
					end
					errs(i) = nnz(ex.Y ~= pred) / ex.nNode;
				end
				trErrs(a,fold,c1,c2) = mean(errs);
				fprintf('Avg train err = %.4f\n', trErrs(a,fold,c1,c2));
				
				%% CROSS-VALIDATION
				
				errs = zeros(nCV,1);
				stab = zeros(nCV,2);
				for i = 1:nCV
					ex = ex_cv{i};
					[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
					if strcmp(algoNames{runAlgos(a)},'SCTSM') ...
					|| strcmp(algoNames{runAlgos(a)},'VCTSM') ...
					|| strcmp(algoNames{runAlgos(a)},'VCTSMlog')
						pred = UGM_Decode_ConvexBP(kappa,nodePot,edgePot,ex.edgeStruct,inferFunc);
					else
						pred = decodeFunc(nodePot,edgePot,ex.edgeStruct);
					end
					errs(i) = nnz(ex.Y ~= pred) / ex.nNode;
				end
				cvErrs(a,fold,c1,c2) = mean(errs);
				fprintf('Avg CV err = %.4f\n', cvErrs(a,fold,c1,c2));
				
				%% TESTING
				
				errs = zeros(nTest,1);
				f1 = zeros(nTest,1);
				for i = 1:nTest
					ex = ex_te{i};
					[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
					if strcmp(algoNames{runAlgos(a)},'SCTSM') ...
					|| strcmp(algoNames{runAlgos(a)},'VCTSM') ...
					|| strcmp(algoNames{runAlgos(a)},'VCTSMlog')
						pred = UGM_Decode_ConvexBP(kappa,nodePot,edgePot,ex.edgeStruct,inferFunc);
					else
						pred = decodeFunc(nodePot,edgePot,ex.edgeStruct);
					end
					errs(i) = nnz(ex.Y ~= pred) / ex.nNode;
					[s.cm,s.err,s.pre,s.rec,s.f1,s.f1avg,s.f1wavg] = errStats(double(ex.Y),pred);
					teStat(a,fold,c1,c2,i) = s;
					if ex.nState == 2
						f1(i) = s.f1(1); % If binary, use F1 of first class
					else
						f1(i) = s.f1wavg; % If multiclass, use weighted average F1
					end
				end
				teErrs(a,fold,c1,c2) = mean(errs);
				teF1(a,fold,c1,c2) = mean(f1);
				fprintf('Avg test err = %.4f, avg F1 = %.4f\n', teErrs(a,fold,c1,c2),teF1(a,fold,c1,c2));
				
				% Plot last prediction
				if isempty(plotFunc)
					if plotPred
						% use default plotting
						figure(plotPred);
						subplot(1,nRunAlgos,a);
						imagesc(reshape(pred,ex.edgeStruct.nRows,ex.edgeStruct.nCols));
						colormap(gray);
						title(sprintf('%s : f=%d, C1=%d, C2=%d', algoNames{runAlgos(a)},fold,c1,c2));
						drawnow;
					end
				else
					plotFunc(pred, ex, expSetup, algoNames{a}, C_w, a);
				end
				
				%% PROGRESS
				
				count = count + 1;
				curTime = toc(totalTimer);
				fprintf('Finished %d of %d; elapsed: %.2f min; ETA: %.2f min\n', ...
					count, nJobs, curTime/60, (nJobs-count)*(curTime/count)/60);
							
			end
			
		end
		
	end
	
	% Choose best parameters
	for a = 1:nRunAlgos
		bestParamTrain(a,fold) = find(trErrs(a,fold,:)==min(trErrs(a,fold,:)),1,'last');
		bestParamCVerr(a,fold) = find(cvErrs(a,fold,:)==min(cvErrs(a,fold,:)),1,'last');
		bestParamTest(a,fold) = find(teErrs(a,fold,:)==min(teErrs(a,fold,:)),1,'last');
	end
	
	% Train on [tr cv]; compute new test stats
	ex_tr_full = [ex_tr ex_cv];
	nTrainFull = length(ex_tr_full);
	for a = 1:nRunAlgos
		% Choose best hyperparams for fold
		[c1idx,c2idx] = ind2sub([nCvals1,nCvals2], bestParamCVerr(a,fold));
		C_w = Cvec(c1idx);
		if strcmp(algoNames{runAlgos(a)},'PERC') || strcmp(algoNames{runAlgos(a)},'M3N')
			stepSize = stepSizeVec(c2idx);
		elseif strcmp(algoNames{runAlgos(a)},'SCTSM')
			kappa = kappaVec(c2idx);
		end
		
		try
		switch(runAlgos(a))

			% M(P)LE learning
			case 1
				fprintf('Training MLE (full) with C=%f \n',C_w);
				%[w,nll] = trainMLE(ex_tr,inferFunc,C_w,optSGD);
				[w,nll] = trainMLE_lbfgs(ex_tr_full,inferFunc,C_w,optVCTSM);
				paramsFull{a,fold}.w = w;

			% Perceptron learning
			case 2
				fprintf('Training Perceptron (full) with C=%f, stepSize=%f \n',C_w,stepSize);
				optM3N.stepSize = stepSize;
				[w,fAvg] = trainPerceptron(ex_tr_full,decodeFunc,C_w,optM3N);
				paramsFull{a,fold}.w = w;

			% M3N learning
			case 3
				fprintf('Training M3N (full) with C=%f, stepSize=%f \n',C_w,stepSize);
				optM3N.stepSize = stepSize;
				[w,fAvg] = trainM3N(ex_tr_full,decodeFunc,C_w,optM3N);
				paramsFull{a,fold}.w = w;

			% M3N learning with FW
			case 4
				fprintf('Training M3NFW (full) with C=%f ...\n',C_w);
				[w,fAvg] = bcfw(ex_tr_full,decodeFunc,C_w,optM3N);
				paramsFull{a,fold}.w = w;

			% SCTSM learning (fixed convexity)
			case 5
				fprintf('Training SCTSM (full) with C=%f, kappa=%f \n',C_w,kappa);
				[w,f] = trainSCTSM(ex_tr_full,inferFunc,kappa,C_w,optSCTSM);
				paramsFull{a,fold}.w = w;

			% VCTSM learning (convexity optimization)
			case 6
				fprintf('Training VCTSM (full) with C=%f ...\n',C_w);
				[w,kappa,f] = trainVCTSM(ex_tr_full,inferFunc,C_w,optVCTSM,[],initKappa);
				paramsFull{a,fold}.w = w;
				paramsFull{a,fold}.kappa = kappa;

			% VCTSM_log learning (log version)
			case 7
				fprintf('Training VCTSMlog (full) with C=%f ...\n',C_w);
				[w,kappa,f] = trainVCTSM_log(ex_tr_full,inferFunc,C_w,optVCTSM,[],initKappa);
				paramsFull{a,fold}.w = w;
				paramsFull{a,fold}.kappa = kappa;

		end

		catch exception % Caught an exception during training
			fprintf('Caught exception : %s\n  Skipping current job.', exception.message);
			continue
		end
		
		% Training stats (full)
		errs = zeros(nTrainFull,1);
		for i = 1:nTrainFull
			ex = ex_tr_full{i};
			[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
			if strcmp(algoNames{runAlgos(a)},'SCTSM') ...
			|| strcmp(algoNames{runAlgos(a)},'VCTSM') ...
			|| strcmp(algoNames{runAlgos(a)},'VCTSMlog')
				pred = UGM_Decode_ConvexBP(kappa,nodePot,edgePot,ex.edgeStruct,inferFunc);
			else
				pred = decodeFunc(nodePot,edgePot,ex.edgeStruct);
			end
			errs(i) = nnz(ex.Y ~= pred) / ex.nNode;
		end
		trErrsFull(a,fold) = mean(errs);
		fprintf('Avg train err = %.4f\n', trErrsFull(a,fold));
		
		% Test stats (full)
		errs = zeros(nTest,1);
		f1 = zeros(nTest,1);
		for i = 1:nTest
			ex = ex_te{i};
			[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
			if strcmp(algoNames{runAlgos(a)},'SCTSM') ...
			|| strcmp(algoNames{runAlgos(a)},'VCTSM') ...
			|| strcmp(algoNames{runAlgos(a)},'VCTSMlog')
				pred = UGM_Decode_ConvexBP(kappa,nodePot,edgePot,ex.edgeStruct,inferFunc);
			else
				pred = decodeFunc(nodePot,edgePot,ex.edgeStruct);
			end
			errs(i) = nnz(ex.Y ~= pred) / ex.nNode;
			[s.cm,s.err,s.pre,s.rec,s.f1,s.f1avg,s.f1wavg] = errStats(double(ex.Y),pred);
			teStatFull(a,fold) = s;
			if ex.nState == 2
				f1(i) = s.f1(1); % If binary, use F1 of first class
			else
				f1(i) = s.f1wavg; % If multiclass, use weighted average F1
			end
		end
		teErrsFull(a,fold) = mean(errs);
		teF1Full(a,fold) = mean(f1);
		fprintf('Avg test err = %.4f, avg F1 = %.4f\n', teErrsFull(a,fold),teF1Full(a,fold));
	end
	
	% Clear old folds (no sense in keeping them)
	clear ex_tr ex_ul ex_cv ex_te;
	
	% Save results
	if ~isempty(save2file)
		save(save2file);
	end
	
	fprintf('\n');
end

endTime = toc(totalTimer);
fprintf('elapsed time: %.2f min\n',endTime/60);

% Generalization error
if useFullTrain
	geErrsFull = teErrsFull - trErrsFull;
else
	geErrs = teErrs - trErrs;
end

% Display results at end
colStr = {'TrainErr','ValidErr','TestErr','TestF1','GenErr','C1','C2','kappa'};
bestParam = bestParamCVerr;
bestResults = zeros(nRunAlgos,length(colStr),nFold);
for fold = 1:nFold
	% Choose best hyperparams for fold
	[c1idx,c2idx] = ind2sub([nCvals1,nCvals2], bestParam(:,fold));
	bestC1 = Cvec(c1idx);
	bestC2 = zeros(nRunAlgos,1);
	bestKappa = zeros(nRunAlgos,1);
	for a = 1:nRunAlgos
		% C2/kappa diplay value
		if strcmp(algoNames{runAlgos(a)},'PERC') || strcmp(algoNames{runAlgos(a)},'M3N')
			bestC2(a) = stepSizeVec(c2idx(a));
		elseif strcmp(algoNames{runAlgos(a)},'SCTSM')
			bestKappa(a) = kappaVec(c2idx(a));
		elseif strcmp(algoNames{runAlgos(a)},'VCTSM') || strcmp(algoNames{runAlgos(a)},'VCTSMlog')
			if useFullTrain
				bestKappa(a) = paramsFull{a,fold}.kappa;
			else
				bestKappa(a) = params{a,fold,bestParam(a,fold)}.kappa;
			end
		end
	end	
	% Best results for fold
	idx = sub2ind(size(teErrs),(1:nRunAlgos)',fold*ones(nRunAlgos,1),c1idx,c2idx);
	if useFullTrain
		bestResults(:,:,fold) = ...
			[trErrsFull(:,fold) cvErrs(idx) teErrsFull(:,fold) teF1Full(:,fold) geErrsFull(:,fold) ...
			bestC1(:) bestC2(:) bestKappa(:)];
	else
		bestResults(:,:,fold) = ...
			[trErrs(idx) cvErrs(idx) teErrs(idx) teF1(idx) geErrs(idx) ...
			bestC1(:) bestC2(:) bestKappa(:)];
	end
end

% Compute mean/stdev across folds
avgResults = mean(bestResults,3);
stdResults = std(bestResults,[],3);

% Paired t-tests
ttests = zeros(nRunAlgos);
sigThresh = 0.05;
for a1 = 1:nRunAlgos
	for a2 = a1+1:nRunAlgos
		ttests(a1,a2) = ttest(squeeze(bestResults(a1,3,:)),squeeze(bestResults(a2,3,:)),sigThresh);
	end
end
ttests(~isfinite(ttests)) = 0;
ttests = ttests | ttests';

% Output results
fprintf('------------\n');
fprintf('FOLD RESULTS\n');
fprintf('------------\n');
for fold = 1:nFold
	if computeBaseline
		fprintf('Fold %d baseline: avg err = %.4f, avg F1 = %.4f\n', fold,baselineErrs(fold),baselineF1(fold));
	end
	disptable(bestResults(:,:,fold),colStr,algoNames(runAlgos),'%.5f');
end
fprintf('-------------\n');
fprintf('FINAL RESULTS\n');
fprintf('-------------\n');
if computeBaseline
	fprintf('Baseline: avg err = %.4f, avg F1 = %.4f\n', mean(baselineErrs),mean(baselineF1));
end
disptable(avgResults,colStr,algoNames(runAlgos),'%.5f');
fprintf('Significance t-tests (threshold=%f)\n',sigThresh);
disptable(ttests,algoNames(runAlgos),algoNames(runAlgos));

% save results
if ~isempty(save2file)
	save(save2file);
end
