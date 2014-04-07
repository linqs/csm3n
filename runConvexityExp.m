%% Experiment to test benefit of convexity
%
% Variables:
%   nFold (def: 10)
%   noiseRate (def: 1)
%	discretize (def: 1)
%   inferFunc (def: UGM_Infer_TRBP)
%   decodeFunc (def: UGM_Decode_TRBP)
%   srcfile (def: will create 7*nFold examples)
%		Must also supply nFold,foldDist
%   filename (def: will not save)

if ~exist('nFold','var')
	nFold = 10;
end
if ~exist('noiseRate','var')
	noiseRate = 1;
end
if ~exist('discretize','var')
	discretize = 1;
end
if ~exist('inferFunc','var')
	inferFunc = @UGM_Infer_TRBP;
end
if ~exist('decodeFunc','var')
	decodeFunc = @UGM_Decode_TRBP;
end

if discretize
	Xdesc = struct('discreteX',1,'nonneg',1);
else
	Xdesc = struct('discreteX',0,'nonneg',0);
end

expSetup = struct('Xdesc',Xdesc,...
				  'runAlgos',5,...
				  'Cvec',[.001 .01 .1 .5 1 5 10 50 100 500 1000],...
				  'kappaVec',[.001 .01 .05 .1 .2 .4 .8 1 1.5 2],...
				  'nStabSamp',0,...
				  'decodeFunc',decodeFunc,'inferFunc',inferFunc);

if ~exist('srcfile','var')
	examples = noisyX(7*nFold,noiseRate,0,discretize,0);
	expSetup.nFold = nFold;
	expSetup.foldDist = [1 0 1 5];
else
	load(srcfile);
	expSetup.nFold = nFold;
	expSetup.foldDist = foldDist;
end

if exist('filename','var')
	expSetup.save2file = filename;
end
% expSetup.optSGD = struct('maxIter',200);
experiment;
