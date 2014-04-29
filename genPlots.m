function genPlots(nPlot,expSetup,teErrs,params,nLocParam,plotFolds,plotCvec,plotKappa)
%
% Plots results
%
% nPlot : 1 plot puts both M3N and VCTSM on same plot; 2 plots separates them.

assert(any(ismember(nPlot,[1 2])), 'nPlot must be either 1 or 2');
	
% Experiment variables
if isfield(expSetup,'foldIdx')
	nFold = length(expSetup.foldIdx);
else
	nFold = expSetup.nFold;
end
Cvec = expSetup.Cvec;
kappaVec = expSetup.kappaVec;
sctsmIdx = find(expSetup.runAlgos==5);

% Which trials to use
if ~exist('plotFolds','var') || isempty(plotFolds)
	plotFolds = 1:nFold;
end
if ~exist('plotCvec','var') || isempty(plotCvec)
	plotCvec = 1:length(Cvec);
end
if ~exist('plotKappa','var') || isempty(plotKappa)
	plotKappa = 1:length(kappaVec);
end

% Compute stats
avgErrC = zeros(length(Cvec),length(kappaVec));
stdErrC = zeros(length(Cvec),length(kappaVec));
avgNormWloc = zeros(length(Cvec),length(kappaVec));
avgNormWrel = zeros(length(Cvec),length(kappaVec));
if length(plotFolds) == 1
	avgErrM3N = squeeze(teErrs(1,plotFolds(1),1:length(Cvec),1))';
	avgErrVCTSM = squeeze(teErrs(2,plotFolds(1),1:length(Cvec),1))';
	for c = 1:length(Cvec)
		avgErrC(c,:) = squeeze(teErrs(sctsmIdx,plotFolds(1),c,1:length(kappaVec)));
		stdErrC(c,:) = 0;
		for k = 1:length(kappaVec)
			avgNormWloc(c,k) = norm(params{sctsmIdx,plotFolds(1),c,k}.w(1:nLocParam))^2;
			avgNormWrel(c,k) = norm(params{sctsmIdx,plotFolds(1),c,k}.w(nLocParam+1:end))^2;
		end
	end
else
	avgErrM3N = mean(squeeze(teErrs(1,plotFolds,1:length(Cvec),1)),1);
	avgErrVCTSM = mean(squeeze(teErrs(2,plotFolds,1:length(Cvec),1)),1);
	for c = 1:length(Cvec)
		avgErrC(c,:) = mean(squeeze(teErrs(sctsmIdx,plotFolds,c,1:length(kappaVec))),1);
		stdErrC(c,:) = std(squeeze(teErrs(sctsmIdx,plotFolds,c,1:length(kappaVec))),1);
		for k = 1:length(kappaVec)
			for f = plotFolds
				avgNormWloc(c,k) = avgNormWloc(c,k) + norm(params{sctsmIdx,f,c,k}.w(1:nLocParam))^2;
				avgNormWrel(c,k) = avgNormWrel(c,k) + norm(params{sctsmIdx,f,c,k}.w(nLocParam+1:end))^2;
			end
			avgNormWloc(c,k) = avgNormWloc(c,k) / length(plotFolds);
		end
	end
end

% Make legend strings
legendStr = {};
for c = 1:length(Cvec)
	legendStr{c} = sprintf('C=%.3f',Cvec(c));
end

if nPlot == 1
	
	% 2 Plots
	fig = figure();
	figPos = get(fig,'Position');
	figPos(3) = 2*figPos(3);
	set(fig,'Position',figPos);

	subplot(1,2,1);
	semilogx(repmat(kappaVec(plotKappa),length(plotCvec),1)',avgErrC(plotCvec,plotKappa)');
	hold on;
	plot(repmat(kappaVec(plotKappa),length(plotCvec),1)',repmat(avgErrM3N(plotCvec),length(plotKappa),1),'-.');
	plot(repmat(kappaVec(plotKappa),length(plotCvec),1)',repmat(avgErrVCTSM(plotCvec),length(plotKappa),1),'--');
	hold off;
	title('Convexity vs. Test Error (dash-dot = M3N, dash-dash = VCTSM)');
	xlabel('log(kappa)'); ylabel(sprintf('test error (avg %d folds)',nFold));
	legend(legendStr(plotCvec),'Location','SouthEast');

	subplot(1,2,2);
	[hAx,hLine1,hLine2] = plotyy(repmat(kappaVec(plotKappa),length(plotCvec),1)',avgNormWloc(plotCvec,plotKappa)',repmat(kappaVec(plotKappa),length(plotCvec),1)',avgNormWrel(plotCvec,plotKappa)');
	set(hLine2,'LineStyle','--');
	title('Convexity vs. Norm of Weights');
	xlabel('kappa');
	ylabel(hAx(1),sprintf('Local ||w||^2 (avg %d folds)',nFold));
	ylabel(hAx(2),sprintf('Relational ||w||^2 (avg %d folds)',nFold));

elseif nPlot == 2
	
	% 3 Plots
	fig = figure();
	figPos = get(fig,'Position');
	figPos(3) = 3*figPos(3);
	set(fig,'Position',figPos);

	subplot(1,3,1);
	semilogx(repmat(kappaVec(plotKappa),length(plotCvec),1)',avgErrC(plotCvec,plotKappa)');
	hold on;
	semilogx(repmat(kappaVec(plotKappa),length(plotCvec),1)',repmat(avgErrM3N(plotCvec),length(plotKappa),1),'-.');
	hold off;
	title('Convexity vs. Test Error (dotted = M3N)');
	xlabel('log(kappa)'); ylabel(sprintf('test error (avg %d folds)',nFold));
	legend(legendStr(plotCvec),'Location','SouthEast');

	subplot(1,3,2);
	semilogx(repmat(kappaVec(plotKappa),length(plotCvec),1)',avgErrC(plotCvec,plotKappa)');
	hold on;
	semilogx(repmat(kappaVec(plotKappa),length(plotCvec),1)',repmat(avgErrVCTSM(plotCvec),length(plotKappa),1),'--');
	hold off;
	title('Convexity vs. Test Error (dotted = VCTSM)');
	xlabel('log(kappa)'); ylabel(sprintf('test error (avg %d folds)',nFold));

	subplot(1,3,3);
	[hAx,hLine1,hLine2] = plotyy(repmat(kappaVec(plotKappa),length(plotCvec),1)',avgNormWloc(plotCvec,plotKappa)',repmat(kappaVec(plotKappa),length(plotCvec),1)',avgNormWrel(plotCvec,plotKappa)');
	set(hLine2,'LineStyle','--');
	title('Convexity vs. Norm of Weights');
	xlabel('kappa');
	ylabel(hAx(1),sprintf('Local ||w||^2 (avg %d folds)',nFold));
	ylabel(hAx(2),sprintf('Relational ||w||^2 (avg %d folds)',nFold));
	
end
