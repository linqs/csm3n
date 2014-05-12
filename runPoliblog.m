%% Poliblog experiment
%
% Variables:
%   noiseRate (def: 0)
%   runAlgos (def: [4 10])
%   inferFunc (def: UGM_Infer_TRBP)
%   decodeFunc (def: UGM_Decode_TRBP)
%   save2file (def: will not save)

if ~exist('noiseRate','var')
	runAlgos = 0;
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

cd data/poliblog/Processed/;
[examples,foldIdx] = loadPoliBlog();
if noiseRate > 0
	examples = perturbExamples(examples,noiseRate);
end
cd ../../../;

maxIter = 200;

expSetup = struct('nFold',1,'foldDist',[1 0 1 2] ...
				 ,'runAlgos',runAlgos ...
				 ,'decodeFunc',@UGM_Decode_TRBP,'inferFunc',@UGM_Infer_TRBP ...
				 ,'Cvec',[.001 .01 .05 .1 .5 1 5 10 50 100] ...
				 ,'Cvec2',[.001 .01 .05 .1 .5 1 5 10 50 100] ...
				 ,'stepSizeVec',[.1 .2 .5 1 2 5 10] ...
				 ,'kappaVec',[.1 .2 .5 1 2 5 10] ...
				 );
expSetup.optSGD = struct('maxIter',maxIter ...
						,'plotObj',0,'plotRefresh',10 ...
						,'verbose',0,'returnBest',1);
expSetup.optLBFGS = struct('Display','off','verbose',0 ...
						  ,'MaxIter',maxIter,'MaxFunEvals',maxIter);
			  
if exist('save2file','var')
	expSetup.save2file = save2file;
end

experiment
