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
if isfield(expSetup,'foldIdx')
	foldIdx = expSetup.foldIdx;
	nFold = length(foldIdx);
elseif isfield(expSetup,'foldDist')
	foldDist = expSetup.foldDist;
	assert(sum(foldDist) <= nEx, 'Number of examples per fold greater than examples.');
	foldIdx = makeFolds(nEx,nFold,foldDist);
	nFold = min(nFold,length(foldIdx));
else
	nExFold = int(nEx / nFold);
	nTrain = 1;
	nUnlab = 1;
	nCV = 1;
	nTest = nExFold - 3;
	foldIdx = makeFolds(nFold,nTrain,nUnlab,nCV,nTest);
	nFold = min(nEx,nFold,length(foldIdx));
end

algoNames = {'MLE','M3N','M3NLRR','VCTSM','CACC','CSM3N','CSCACC','DLM'};
if isfield(expSetup,'runAlgos')
	runAlgos = expSetup.runAlgos;
else
	runAlgos = 1:length(algoNames);
end
nRunAlgos = length(runAlgos);

if isfield(expSetup,'Cvec')
	Cvec = expSetup.Cvec;
else
	Cvec = [0; reshape([.5*10.^linspace(0,4,5);10.^linspace(0,4,5)],[],1)];
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
nCvals1 = length(Cvec);
nCvals2 = max(length(CvecRel),length(CvecStab));

if isfield(expSetup,'decodeFunc')
	decodeFunc = expSetup.decodeFunc;
else
	decodeFunc = @UGM_Decode_LBP;
end
if isfield(expSetup,'inferFunc')
	inferFunc = expSetup.inferFunc;
else
	inferFunc = @UGM_Infer_LBP;
end
if isfield(expSetup,'edgeFeatFunc')
	edgeFeatFunc = expSetup.edgeFeatFunc;
else
	edgeFeatFunc = @makeEdgeFeatures;
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

if isfield(expSetup,'save2file')
	save2file = expSetup.save2file;
else
	save2file = [];
end


%% MAIN LOOP

% job metadata
nJobs = nFold * nCvals1 * (...
	length(intersect(runAlgos,[1 2 4 5 8])) + ...
	length(CvecRel) * nnz(runAlgos==3) + ...
	length(CvecStab) * length(intersect(runAlgos,[6 7])) );
totalTimer = tic;
count = 0;

% storage
params = cell(nRunAlgos,nFold,nCvals1,nCvals2);
trErrs = inf(nRunAlgos,nFold,nCvals1,nCvals2);
cvErrs = inf(nRunAlgos,nFold,nCvals1,nCvals2);
teErrs = inf(nRunAlgos,nFold,nCvals1,nCvals2);
perturbs = cell(nFold,1);
cvStabMax = inf(nRunAlgos,nFold,nCvals1,nCvals2,2);
cvStabAvg = inf(nRunAlgos,nFold,nCvals1,nCvals2,2);

% best parameters based on {CV,stab,test)
bestParamCVerr = zeros(nRunAlgos,nFold);
bestParamStab = zeros(nRunAlgos,nFold);
bestParamTest = zeros(nRunAlgos,nFold);

for fold = 1:nFold
	
	fprintf('Starting fold %d of %d.\n', fold,nFold);
	
	% separate training/CV/testing
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
	
	% init perturbations for stability measurement
	pert = [];
	
	for c1 = 1:nCvals1
		C_w = Cvec(c1);
		
		for c2 = 1:nCvals2
			
			for a = 1:nRunAlgos

				if (nnz(runAlgos(a) == [3 6 7]) == 0) && c2 > 1
					% Some algorithms don't use second reg. param.
					continue;
				elseif runAlgos(a) == 3
					% M3NLRR
					if c2 > length(CvecRel)
						continue;
					end
					C_r = CvecRel(c2);
				elseif runAlgos(a) == 6 || runAlgos(a) == 7
					% CSM3N or CSCACC
					if c2 > length(CvecStab)
						continue;
					end
					C_s = CvecStab(c2);
				end
				
				%% TRAINING

				switch(runAlgos(a))

					% M(P)LE learning
					case 1
						fprintf('Training MLE ...\n');
						[w,nll] = trainMLE(ex_tr,inferFunc,C_w);
						params{a,fold,c1,c2}.w = w;

					% M3N learning
					case 2
						fprintf('Training M3N ...\n');
						[w,fAvg] = trainM3N(ex_tr,decodeFunc,C_w);
						params{a,fold,c1,c2}.w = w;

					% M3NLRR learning (M3N with separate local/relational reg.)
					case 3
						fprintf('Training M3N with local/relational regularization ...\n');
						maxLocParamIdx = max(ex_tr{1}.nodeMap(:));
						Csplit = [C_w * ones(maxLocParamIdx,1) ; C_r * ones(nParam-maxLocParamIdx,1)];
						[w,fAvg] = trainM3N(ex_tr,decodeFunc,Csplit);
						params{a,fold,c1,c2}.w = w;

					% VCTSM learning (convexity optimization)
					case 4
						fprintf('Training VCTSM ...\n');
						[w,kappa,f] = trainVCTSM(ex_tr,C_w);
						params{a,fold,c1,c2}.w = w;
						params{a,fold,c1,c2}.kappa = kappa;

					% CACC learning (robust M3N)
					case 5
						fprintf('Training CACC ...\n');
						[w,fAvg] = trainCACC(ex_tr,decodeFunc,C_w);
						params{a,fold,c1,c2}.w = w;

					% CSM3N learning (M3N + stability reg.)
					case 6
						fprintf('Training CSM3N ...\n');
						[w,fAvg] = trainCSM3N(ex_tr,ex_ul,decodeFunc,C_w,C_s);
						params{a,c1,fold}.w = w;

					% CSCACC learning (CACC + stability reg.)
					case 7
						fprintf('Training CSCACC ...\n');
						[w,fAvg] = trainCSCACC(ex_tr,ex_ul,decodeFunc,C_w,C_s);
						params{a,fold,c1,c2}.w = w;

					% DLM learning
					case 8
						fprintf('Training DLM ...\n');
						[w,fAvg] = trainDLM(ex_tr,decodeFunc,C_w);
						params{a,fold,c1,c2}.w = w;

				end

				% training stats
				errs = zeros(nTrain,1);
				for i = 1:nTrain
					ex = ex_tr{i};
					if a ~= 4
						[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
						pred = decodeFunc(nodePot,edgePot,ex.edgeStruct);
					else
						mu = vctsmInfer(w,kappa,ex.Fx,ex.Aeq,ex.beq);
						pred = decodeMarginals(mu,ex.nNode,ex.nState);
					end
					errs(i) = nnz(ex.Y ~= pred) / ex.nNode;
				end
				trErrs(a,fold,c1,c2) = sum(errs)/nTrain;
				fprintf('Avg train err = %.4f\n', trErrs(a,fold,c1,c2));

				%% CROSS-VALIDATION

				errs = zeros(nCV,1);
				stab = zeros(nCV,2);
				for i = 1:nCV
					ex = ex_cv{i};
					if a ~= 4
						[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
						pred = decodeFunc(nodePot,edgePot,ex.edgeStruct);
						if nStabSamp > 0
							[stab(i,1),stab(i,2),pert] = measureStabilityRand({w},ex,discreteX,nStabSamp,decodeFunc,edgeFeatFunc,pred,pert);
						end
					else
						mu = vctsmInfer(w,kappa,ex.Fx,ex.Aeq,ex.beq);
						pred = decodeMarginals(mu,ex.nNode,ex.nState);
						if nStabSamp > 0
							[stab(i,1),stab(i,2),pert] = measureStabilityRand({w,kappa},ex,discreteX,nStabSamp,[],edgeFeatFunc,pred,pert);
						end
					end
					errs(i) = nnz(ex.Y ~= pred()) / ex.nNode;
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
				for i = 1:nTest
					ex = ex_te{i};
					if a ~= 4
						[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
						pred = decodeFunc(nodePot,edgePot,ex.edgeStruct);
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
				teErrs(a,fold,c1,c2) = sum(errs)/nTest;
				fprintf('Avg test err = %.4f\n', teErrs(a,fold,c1,c2));

				%% PROGRESS

				count = count + 1;
				curTime = toc(totalTimer);
				fprintf('Finished %d of %d; elapsed: %.2f min; ETA: %.2f min\n', ...
					count, nJobs, curTime/60, (nJobs-count)*(curTime/count)/60);
				
			end

		end
		
	end
	
	% choose best parameters
	for a = 1:nRunAlgos
		bestParamCVerr(a,fold) = find(cvErrs(a,fold,:)==min(cvErrs(a,fold,:)),1,'first');
		bestParamStab(a,fold) = find(cvStabMax(a,fold,:)==min(cvStabMax(a,fold,:)),1,'first');
		bestParamTest(a,fold) = find(teErrs(a,fold,:)==min(teErrs(a,fold,:)),1,'first');
	end
	
	% store perturbations
	perturbs{fold} = pert;
	
	% save results
	if ~isempty(save2file)
		save(save2file);
	end

	fprintf('\n');

end

% generalization error
geErrs = teErrs - trErrs;

% display results at end
colStr = {'Train','Valid','Test','GenErr','MaxStab','AvgStab','C_w','C_r/C_s'};
bestParam = bestParamCVerr;
bestResults = zeros(nRunAlgos,length(colStr),nFold);
for fold = 1:nFold
	idx = sub2ind([nRunAlgos nFold nCvals1 nCvals2],(1:nRunAlgos)',fold*ones(nRunAlgos,1),bestParam(:,fold));
	[c1idx,c2idx] = ind2sub([nCvals1,nCvals2], bestParam(:,fold));
	bestC1 = Cvec(c1idx);
	bestC2 = zeros(nRunAlgos,1);
	for a = 1:nRunAlgos
		if runAlgos(a) == 3
			bestC2(a) = CvecRel(c2idx(a));
		elseif runAlgos(a) == 6 || runAlgos(a) == 7
			bestC2(a) = CvecStab(c2idx(a));
		end
	end
	bestResults(:,:,fold) = [trErrs(idx) cvErrs(idx) teErrs(idx) geErrs(idx) cvStabMax(idx) cvStabAvg(idx) bestC1(:) bestC2(:)];
	disptable(bestResults(:,:,fold),colStr,algoNames(runAlgos),'%.5f');
end

% compute mean/stdev across folds
avgResults = mean(bestResults,3);
stdResults = std(bestResults,[],3);



