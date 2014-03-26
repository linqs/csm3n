%% Experiment to test benefit of convexity
%
% Variables:
%   nFold (def: 10)
%   noiseRate (def: 2)
%   inferFunc (def: UGM_Infer_TRBP)
%   decodeFunc (def: UGM_Decode_TRBP)
%   filename (def: will not save)

if ~exist('nFold','var')
	nFold = 10;
end
if ~exist('noiseRate','var')
	noiseRate = 2;
end
if ~exist('inferFunc','var')
	inferFunc = @UGM_Infer_TRBP;
end
if ~exist('decodeFunc','var')
	decodeFunc = @UGM_Decode_TRBP;
end
examples = noisyX(3*nFold,noiseRate,0,1,0);
Xdesc = struct('discreteX',1,'nonneg',1);
expSetup = struct('Xdesc',Xdesc,...
				  'nFold',nFold,'foldDist',[1 0 1 1],...
				  'runAlgos',5,...
				  'Cvec',[.001 .01 .1 .5 1 5 10 50 100 500 1000],...
				  'kappaVec',[.001 .01 .05 .1 .2 .4 .8 1 2],...
				  'nStabSamp',0,...
				  'decodeFunc',decodeFunc,'inferFunc',inferFunc);
if exist('filename','var')
	expSetup.save2file = filename;
end
% expSetup.optSGD = struct('maxIter',200);
experiment;
