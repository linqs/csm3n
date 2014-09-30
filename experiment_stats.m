% Variables:
%   useFullTrain (def: 0)
%   sigThresh (def: 0.05)

if ~exist('useFullTrain','var')
	useFullTrain = 0;
end
if ~exist('sigThresh','var')
	sigThresh = 0.05;
end

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
