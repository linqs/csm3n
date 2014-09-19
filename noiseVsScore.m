function noiseVsScore(filenames,noiseRates,scoreIdx,scoreName,plotErrBars)

if ~exist('plotErrBars','var') || isempty(plotErrBars)
	plotErrBars = 0;
end

algoNames = {'MM','VCMM','SCMM'};
m3nIdx = 3;
vctsmIdx = 1;
sctsmIdx = 2;

errs = zeros(3,length(noiseRates));
stds = zeros(3,length(noiseRates));
% kappas = zeros(1,length(noiseRates));
for f = 1:length(filenames)
	load(filenames{f},'avgResults','stdResults','nFold','bestParam','params');
	errs(:,f) = avgResults(:,scoreIdx);
	stds(:,f) = stdResults(:,scoreIdx);
% 	for fold = 1:nFold
% 		kappas(f) = kappas(f) + params{2,fold,bestParam(2,fold)}.kappa;
% 	end
% 	kappas(f) = kappas(f) / nFold;
end

fontSize = 24;

figure();
if plotErrBars
	errorbar(noiseRates,errs(m3nIdx,:),stds(1,:),'rx-.','MarkerSize',16,'LineWidth',4);
	hold on;
	errorbar(noiseRates,errs(vctsmIdx,:),stds(2,:),'bo--','MarkerSize',10,'LineWidth',4);
% 	errorbar(noiseRates,errs(sctsmIdx,:),stds(3,:),'g--','MarkerSize',14,'LineWidth',4);
else
	plot(noiseRates,errs(m3nIdx,:),'rx-.','MarkerSize',16,'LineWidth',4);
	hold on;
	plot(noiseRates,errs(vctsmIdx,:),'bo--','MarkerSize',10,'LineWidth',4);
% 	plot(noiseRates,errs(sctsmIdx,:),'g--','MarkerSize',14,'LineWidth',4);
end
leg = legend('MM','VCMM','SCMM','Location','SouthEast');
set(leg,'FontSize',fontSize);
set(gca,'box','off','FontSize',fontSize-6);
xlabel('noise rate','FontSize',fontSize);
ylabel(sprintf('%s (avg %d folds)',scoreName,nFold),'FontSize',fontSize);
axis tight;
hold off;

% figure();
% plot(noiseRates,errs(m3nIdx,:)-errs(vctsmIdx,:),'LineWidth',4);
