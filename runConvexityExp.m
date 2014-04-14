%% Experiment to test benefit of convexity
%
% Variables:
%   nFold (def: 10)
%   noiseRate (def: 1)
%   discretize (def: 1)
%   inferFunc (def: UGM_Infer_TRBP)
%   decodeFunc (def: UGM_Decode_TRBP)
%   srcfile (def: will create 7*nFold examples)
%		Must also supply either foldIdx or nFold,foldDist
%   savefile (def: will not save)

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
if ~exist('Cvec','var')
	Cvec = [.001 .01 .1 .5 1 5 10 50 100 500 1000];
end
if ~exist('kappaVec','var')
	kappaVec = [.001 .01 .02 .05 .1 .25 .5 .75 1 1.5 2];
end

if discretize
	Xdesc = struct('discreteX',1,'nonneg',1);
else
	Xdesc = struct('discreteX',0,'nonneg',0);
end

expSetup = struct('Xdesc',Xdesc,...
				  'runAlgos',[2 4 5],...
				  'Cvec',Cvec,...
				  'kappaVec',kappaVec,...
				  'nStabSamp',0,...
				  'decodeFunc',decodeFunc,'inferFunc',inferFunc);

if ~exist('srcfile','var')
	examples = noisyX(7*nFold,noiseRate,0,discretize,0);
	expSetup.nFold = nFold;
	expSetup.foldDist = [1 0 1 5];
else
	load(srcfile);
	if exist('foldIdx','var')
		expSetup.foldIdx = foldIdx;
	else
		expSetup.nFold = nFold;
		expSetup.foldDist = foldDist;
	end
end

if exist('savefile','var')
	expSetup.save2file = savefile;
end

if exist('optSGD','var')
	expSetup.optSGD = optSGD;
end

experiment;
