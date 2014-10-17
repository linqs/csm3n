function [results,algos] = combineExperiments(fnames,algonames,sigThresh,ttestMetricIdx)

if ~exist('sigThresh','var') || isempty(sigThresh)
	sigThresh = 0.05;
end
if ~exist('ttestMetricIdx','var') || isempty(ttestMetricIdx)
	ttestMetricIdx = 3;
end

colStr = {'TrainErr','ValidErr','TestErr','TestF1','GenErr','C1','C2','kappa'};

% Combine best results from all experiments
algos = {};
results = [];
for f = 1:length(fnames)
	fn = fnames{f};
	load(fn,'expSetup','bestResults');
	algos = [algos algonames{f}];
	results = [results ; bestResults];
end

% Compute mean/stdev across folds
avgResults = mean(results,3);
stdResults = std(results,[],3);

% Paired t-tests
ttests = zeros(length(algos));
for a1 = 1:length(algos)
	for a2 = a1+1:length(algos)
		ttests(a1,a2) = ttest(squeeze(results(a1,ttestMetricIdx,:)),squeeze(results(a2,ttestMetricIdx,:)),sigThresh);
	end
end
ttests(~isfinite(ttests)) = 0;
ttests = ttests | ttests';

% Output results
fprintf('------------\n');
fprintf('FOLD RESULTS\n');
fprintf('------------\n');
for fold = 1:size(results,2)
	disptable(results(:,:,fold),colStr,algos,'%.5f');
end
fprintf('-------------\n');
fprintf('FINAL RESULTS\n');
fprintf('-------------\n');
disptable(avgResults,colStr,algos,'%.5f');
fprintf('Significance t-tests (threshold=%f)\n',sigThresh);
disptable(ttests,algos,algos);

