%% Discrete noisyX experiment
%
% Variables:
%   nFold (def: 10)
%   noiseRate (def: 2)
%   runAlgos (def: [1 2 4])
%   inferFunc (def: UGM_Infer_LBP)
%   decodeFunc (def: UGM_Decode_LBP)
%   filename (def: will not save)

if ~exist('nFold','var')
	nFold = 5;
end
if ~exist('noiseRate','var')
	noiseRate = 2;
end
if ~exist('runAlgos','var')
	runAlgos = [1 2 4];
end
if ~exist('inferFunc','var')
	inferFunc = @UGM_Infer_LBP;
end
if ~exist('decodeFunc','var')
	decodeFunc = @UGM_Decode_LBP;
end
examples = noisyX(4*nFold,noiseRate,0,1,0);
Xdesc = struct('discreteX',1,'nonneg',1);
expSetup = struct('Xdesc',Xdesc,...
				  'nFold',nFold,'foldDist',[1 1 1 1],...
				  'runAlgos',runAlgos,...
				  'Cvec',[.001 .01 .1 .5 1 5 10 50 100 500 1000],...
				  'nStabSamp',0,...
				  'decodeFunc',decodeFunc,'inferFunc',inferFunc);
if exist('filename','var')
	expSetup.save2file = filename;
end
% expSetup.optSGD = struct('maxIter',200);
experiment;
