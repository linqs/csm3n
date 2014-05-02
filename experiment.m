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

algoNames = {'MLE','M3N','M3NLRR','VCTSM','SCTSM','CACC','CSM3N','CSCACC','DLM'};
if isfield(expSetup,'runAlgos')
	runAlgos = expSetup.runAlgos;
else
	runAlgos = 1:length(algoNames);
end
nRunAlgos = length(runAlgos);

if isfield(expSetup,'Cvec')
	Cvec = expSetup.Cvec;
else
	Cvec = [0 .1 .5 1 5 10 50 100 500 1000 5000 10000];
end
if isfield(expSetup,'CvecRel')
	CvecRel = expSetup.CvecRel;
else
	CvecRel = Cvec;
end
if isfield(expSetup,'CvecStab')
	CvecStab = expSetup.CvecStab;
else
	CvecStab = [.01 .05 .1 .25 .5 .75 1 2];
end
if isfield(expSetup,'kappaVec')
	kappaVec = expSetup.kappaVec;
else
	kappaVec = [.01 .1 .25 .5 1 2];
end
nCvals1 = length(Cvec);
nCvals2 = max([length(CvecRel),length(CvecStab),length(kappaVec)]);

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

if isfield(expSetup,'Xdesc')
	Xdesc = expSetup.Xdesc;
else
	Xdesc = [];
end

if isfield(expSetup,'nStabSamp')
	nStabSamp = expSetup.nStabSamp;
else
	nStabSamp = 10;
end

if isfield(expSetup,'save2file')
	save2file = expSetup.save2file;
else
	save2file = [];
end

if isfield(expSetup,'optSGD')
	optSGD = expSetup.optSGD;
else
	optSGD = struct();
end

if isfield(expSetup,'optSGD_M3N')
	optSGD_M3N = expSetup.optSGD_M3N;
else
	optSGD_M3N = optSGD;
end
if isfield(expSetup,'optSGD_VCTSM')
	optSGD_VCTSM = expSetup.optSGD_VCTSM;
else
	optSGD_VCTSM = optSGD;
end
if isfield(expSetup,'optSGD_CACC')
	optSGD_CACC = expSetup.optSGD_CACC;
else
	optSGD_CACC = optSGD;
end

if isfield(expSetup,'initKappa')
	initKappa = expSetup.initKappa;
else
	initKappa = 1;
end

if isfield(expSetup,'plotPred')
	plotPred = expSetup.plotPred;
else
	plotPred = 0;
end


%% MAIN LOOP

% Job metadata
nJobs = nFold * nCvals1 * (...
	length(intersect(runAlgos,[1 2 4 6 9])) + ...
	length(CvecRel) * any(runAlgos==3) + ...
	length(kappaVec) * any(runAlgos==5) + ...
	length(CvecStab) * length(intersect(runAlgos,[7 8])) );
totalTimer = tic;
count = 0;

% Storage
params = cell(nRunAlgos,nFold,nCvals1,nCvals2);
trErrs = inf(nRunAlgos,nFold,nCvals1,nCvals2);
cvErrs = inf(nRunAlgos,nFold,nCvals1,nCvals2);
teErrs = inf(nRunAlgos,nFold,nCvals1,nCvals2);
teF1 = inf(nRunAlgos,nFold,nCvals1,nCvals2);
perturbs = cell(nFold,1);
cvStabMax = inf(nRunAlgos,nFold,nCvals1,nCvals2,2);
cvStabAvg = inf(nRunAlgos,nFold,nCvals1,nCvals2,2);

% Best parameters based on {CV,stab,test)
bestParamCVerr = zeros(nRunAlgos,nFold);
bestParamStab = zeros(nRunAlgos,nFold);
bestParamTest = zeros(nRunAlgos,nFold);

% Number of local parameters
nLocParam = max(examples{1}.nodeMap(:));

% % Setup figure for plotting predictions
% if plotPred
% 	fig = figure(plotPred);
% 	figpos = get(fig,'Position');
% 	figpos(4) = nRunAlgos*figpos(4);
% 	set(fig,'Position',figpos);
% 	drawnow;
% end

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
	
	for c1 = 1:nCvals1
		C_w = Cvec(c1);
		
		for c2 = 1:nCvals2
			
			for a = 1:nRunAlgos

				if strcmp(algoNames{runAlgos(a)},'M3NLRR')
					% M3NLRR
					if c2 > length(CvecRel)
						continue;
					end
					C_r = CvecRel(c2);
				elseif strcmp(algoNames{runAlgos(a)},'SCTSM')
					% SCTSM
					if c2 > length(kappaVec)
						continue;
					end
					kappa = kappaVec(c2);
				elseif strcmp(algoNames{runAlgos(a)},'CSM3N') || strcmp(algoNames{runAlgos(a)},'CSCACC')
					% CSM3N or CSCACC
					if c2 > length(CvecStab)
						continue;
					end
					C_s = CvecStab(c2);
				elseif c2 > 1
					% Some algorithms don't use second reg. param.
					continue;
				end
				
				%% TRAINING

				switch(runAlgos(a))

					% M(P)LE learning
					case 1
						fprintf('Training MLE ...\n');
						%[w,nll] = trainMLE(ex_tr,inferFunc,C_w,optSGD);
						[w,nll] = trainMLE_lbfgs(ex_tr,inferFunc,C_w);
						params{a,fold,c1,c2}.w = w;

					% M3N learning
					case 2
						fprintf('Training M3N ...\n');
						[w,fAvg] = trainM3N(ex_tr,decodeFunc,C_w,optSGD_M3N);
						params{a,fold,c1,c2}.w = w;

					% M3NLRR learning (M3N with separate local/relational reg.)
					case 3
						fprintf('Training M3N with local/relational regularization ...\n');
						Csplit = [C_w * ones(nLocParam,1) ; C_r * ones(nParam-nLocParam,1)];
						[w,fAvg] = trainM3N(ex_tr,decodeFunc,Csplit,optSGD_M3N);
						params{a,fold,c1,c2}.w = w;

					% VCTSM learning (convexity optimization)
					case 4
						fprintf('Training VCTSM ...\n');
						[w,kappa,f] = trainVCTSM(ex_tr,inferFunc,C_w,1,optSGD_VCTSM,[],initKappa);
						%[w,kappa,f] = trainVCTSM_lbfgs(ex_tr,inferFunc,C_w,1,[],[],initKappa);
						params{a,fold,c1,c2}.w = w;
						params{a,fold,c1,c2}.kappa = kappa;
						
					% SCTSM learning (fixed convexity)
					case 5
						fprintf('Training SCTSM with kappa=%f ...\n',kappa);
						[w,f] = trainSCTSM(ex_tr,inferFunc,kappa,C_w,optSGD_M3N);
						params{a,fold,c1,c2}.w = w;

					% CACC learning (robust M3N)
					case 6
						fprintf('Training CACC ...\n');
						[w,fAvg] = trainCACC(ex_tr,decodeFunc,C_w,optSGD_CACC);
						params{a,fold,c1,c2}.w = w;

					% CSM3N learning (M3N + stability reg.)
					case 7
						fprintf('Training CSM3N ...\n');
						[w,fAvg] = trainCSM3N(ex_tr,ex_ul,decodeFunc,C_w,C_s,optSGD_CACC);
						params{a,fold,c1,c2}.w = w;

					% CSCACC learning (CACC + stability reg.)
					case 8
						fprintf('Training CSCACC ...\n');
						[w,fAvg] = trainCSCACC(ex_tr,ex_ul,decodeFunc,C_w,C_s,optSGD_CACC);
						params{a,fold,c1,c2}.w = w;

					% DLM learning
					case 9
						fprintf('Training DLM ...\n');
						[w,fAvg] = trainDLM(ex_tr,decodeFunc,C_w,optSGD);
						params{a,fold,c1,c2}.w = w;

				end

				% Training stats
				errs = zeros(nTrain,1);
				for i = 1:nTrain
					ex = ex_tr{i};
					[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
					if strcmp(runAlgos(a),'VCTSM') || strcmp(runAlgos(a),'SCTSM')
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
					if strcmp(runAlgos(a),'VCTSM') || strcmp(runAlgos(a),'SCTSM')
						pred = UGM_Decode_ConvexBP(kappa,nodePot,edgePot,ex.edgeStruct,inferFunc);
						if nStabSamp > 0
							sctsmDecoder = @(nodePot,edgePot,edgeStruct) UGM_Decode_ConvexBP(kappa,nodePot,edgePot,edgeStruct,inferFunc);
							[stab(i,1),stab(i,2),pert] = measureStabilityRand({w},ex,Xdesc,nStabSamp,sctsmDecoder,edgeFeatFunc,pred,pert);
						end
					else
						pred = decodeFunc(nodePot,edgePot,ex.edgeStruct);
						if nStabSamp > 0
							[stab(i,1),stab(i,2),pert] = measureStabilityRand({w},ex,Xdesc,nStabSamp,decodeFunc,edgeFeatFunc,pred,pert);
						end
					end
					errs(i) = nnz(ex.Y ~= pred) / ex.nNode;
				end
				cvErrs(a,fold,c1,c2) = mean(errs);
				fprintf('Avg CV err = %.4f\n', cvErrs(a,fold,c1,c2));
				if nStabSamp > 0
					cvStabMax(a,fold,c1,c2) = max(stab(:,1));
					cvStabAvg(a,fold,c1,c2) = mean(stab(:,2));
					fprintf('CV stab: max = %d, avg = %.4f\n', cvStabMax(a,fold,c1,c2),cvStabAvg(a,fold,c1,c2));
				end

				%% TESTING

				errs = zeros(nTest,1);
				f1 = zeros(nTest,1);
				for i = 1:nTest
					ex = ex_te{i};
					[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
					if strcmp(runAlgos(a),'VCTSM') || strcmp(runAlgos(a),'SCTSM')
						pred = UGM_Decode_ConvexBP(kappa,nodePot,edgePot,ex.edgeStruct,inferFunc);
					else
						pred = decodeFunc(nodePot,edgePot,ex.edgeStruct);						
					end
					errs(i) = nnz(ex.Y ~= pred) / ex.nNode;
					[s.cm,s.err,s.pre,s.rec,s.f1,s.f1avg,s.f1wavg] = errStats(double(ex.Y),pred);
					teStat(a,fold,c1,c2,i) = s;
					if ex.nState == 2
						% If binary, use F1 of first class
						f1(i) = s.f1(1);
					else
						% If multiclass, use weighted average F1
						f1(i) = s.f1w;
					end
				end
				teErrs(a,fold,c1,c2) = mean(errs);
				teF1(a,fold,c1,c2) = mean(f1);
				fprintf('Avg test err = %.4f, avg F1 = %.4f\n', teErrs(a,fold,c1,c2),teF1(a,fold,c1,c2));

				% Plot last prediction
				if plotPred
					figure(plotPred);
					subplot(nRunAlgos,1,a);
					imagesc(reshape(pred,ex.edgeStruct.nRows,ex.edgeStruct.nCols));
					colormap(gray);
					title(sprintf('%s : f=%d, C1=%d, C2=%d', algoNames{runAlgos(a)},fold,c1,c2));
					drawnow;
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
		bestParamCVerr(a,fold) = find(cvErrs(a,fold,:)==min(cvErrs(a,fold,:)),1,'first');
		bestParamStab(a,fold) = find(cvStabMax(a,fold,:)==min(cvStabMax(a,fold,:)),1,'first');
		bestParamTest(a,fold) = find(teErrs(a,fold,:)==min(teErrs(a,fold,:)),1,'first');
	end
	
	% Store perturbations
	perturbs{fold} = pert;
	
	% Clear old folds (no sense in keeping them)
	clear ex_tr ex_ul ex_cv ex_te;
	
	% Save results
	if ~isempty(save2file)
		save(save2file);
	end

	fprintf('\n');

end

% Generalization error
geErrs = teErrs - trErrs;

% choose which "best parameters" to use
% bestParam = zeros(nRunAlgos,nFold);
% for a = 1:nRunAlgos
% 	for fold = 1:nFold
% 		if ismember(runAlgos(a),[1 2 6 9])
% 			% algos with only 1 weight regularizer
% 			bestParam(a,fold) = find(cvErrs(a,fold,:)==min(cvErrs(a,fold,:)),1,'first');
% % 			bestParam(a,fold) = find(cvErrs(a,fold,2:end,1)==min(cvErrs(a,fold,2:end,1)),1,'first') + 1;
% 		elseif runAlgos(a) == 3
% 			% M3NLRR
% 			bestParam(a,fold) = find(cvErrs(a,fold,:)==min(cvErrs(a,fold,:)),1,'first');
% 		elseif runAlgos(a) == 4
% 			% VCTSM
% 			% must use some regularization
% 			bestParam(a,fold) = find(cvErrs(a,fold,2:end,1)==min(cvErrs(a,fold,2:end,1)),1,'first') + 1;
% 		else
% 			% CS algos
% 			bestParam(a,fold) = find(cvErrs(a,fold,1,:)==min(cvErrs(a,fold,1,:)),1,'first')*nCvals2 + 1;
% 		end
% 	end
% end
% or just use whatever is best according to CV error
bestParam = bestParamCVerr;

% display results at end
colStr = {'TrainErr','ValidErr','TestErr','TestF1','GenErr','MaxStab','AvgStab','C_w','[C_r|C_s|kappa]'};
bestResults = zeros(nRunAlgos,length(colStr),nFold);
for fold = 1:nFold
	% choose best params for fold
	[c1idx,c2idx] = ind2sub([nCvals1,nCvals2], bestParam(:,fold));
	bestC1 = Cvec(c1idx);
	bestC2 = zeros(nRunAlgos,1);
	for a = 1:nRunAlgos
		if strcmp(algoNames{runAlgos(a)},'M3NLRR')
			bestC2(a) = CvecRel(c2idx(a));
		elseif strcmp(algoNames{runAlgos(a)},'SCTSM')
			bestC2(a) = kappaVec(c2idx(a));
		elseif strcmp(algoNames{runAlgos(a)},'CSM3N') || strcmp(algoNames{runAlgos(a)},'CSCACC')
			bestC2(a) = CvecStab(c2idx(a));
		end
	end
	% display results for fold
	idx = sub2ind([nRunAlgos nFold nCvals1 nCvals2],(1:nRunAlgos)',fold*ones(nRunAlgos,1),c1idx,c2idx);
	bestResults(:,:,fold) = [trErrs(idx) cvErrs(idx) teErrs(idx) teF1(idx) geErrs(idx) ...
							 cvStabMax(idx) cvStabAvg(idx) bestC1(:) bestC2(:)];
	disptable(bestResults(:,:,fold),colStr,algoNames(runAlgos),'%.5f');
end

% compute mean/stdev across folds
avgResults = mean(bestResults,3);
stdResults = std(bestResults,[],3);

endTime = toc(totalTimer);
fprintf('-------------\n');
fprintf('FINAL RESULTS\n');
fprintf('-------------\n');
disptable(avgResults,colStr,algoNames(runAlgos),'%.5f');
fprintf('elapsed time: %.2f min\n',endTime/60);

% save results
if ~isempty(save2file)
	save(save2file);
end
