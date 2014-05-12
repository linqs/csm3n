%% Cora experiment
%
% Variables:
%   nPC (def: 20)
%   runAlgos (def: [4 10])
%   inferFunc (def: UGM_Infer_TRBP)
%   decodeFunc (def: UGM_Decode_TRBP)
%   save2file (def: will not save)
%	makePlots (def: 0)

if ~exist('nPC','var')
	nPC = 20;
end
if ~exist('runAlgos','var')
	runAlgos = [4 10];
end
if ~exist('inferFunc','var')
	inferFunc = @UGM_Infer_TRBP;
end
if ~exist('decodeFunc','var')
	decodeFunc = @UGM_Decode_TRBP;
end

cd data;
[examples,foldIdx] = loadDocData('cora/cora.mat',3,{1:758,759:1758,1759:2708},nPC,1,0);
cd ..;

maxIter = 200;

expSetup = struct('foldIdx',foldIdx ...
				 ,'runAlgos',runAlgos ...
				 ,'decodeFunc',@UGM_Decode_TRBP,'inferFunc',@UGM_Infer_TRBP ...
				 ,'Cvec',[.001 .01 .1 1 10] ...
				 ,'Cvec2',[.001 .01 .1 1 10] ...
				 ,'stepSizeVec',[.1 .2 .5 1 2 5 10] ...
				 ,'kappaVec',[.1 .2 .5 1 2 5 10] ...
				 ,'computeBaseline',1 ...
				 );
expSetup.optSGD = struct('maxIter',maxIter ...
						,'plotObj',objFig,'plotRefresh',10 ...
						,'verbose',0,'returnBest',1);
expSetup.optLBFGS = struct('Display','off','verbose',0 ...
						  ,'MaxIter',maxIter,'MaxFunEvals',maxIter);
expSetup.plotPred = predFig;
			  
if exist('save2file','var')
	expSetup.save2file = save2file;
end

experiment;

