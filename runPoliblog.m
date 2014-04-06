%% Poliblog experiment
%
% Variables:
%   runAlgos (def: [1 2 4])
%   inferFunc (def: UGM_Infer_TRBP)
%   decodeFunc (def: UGM_Decode_TRBP)
%   srcfile (def: will load data)
%   filename (def: will not save)

if ~exist('runAlgos','var')
	runAlgos = [1 2 4];
end
if ~exist('inferFunc','var')
	inferFunc = @UGM_Infer_TRBP;
end
if ~exist('decodeFunc','var')
	decodeFunc = @UGM_Decode_TRBP;
end

Xdesc = struct('discreteX',1,'nonneg',1);
expSetup = struct('Xdesc',Xdesc,...
				  'runAlgos',runAlgos,...
				  'Cvec',[.001 .01 .1 .5 1 5 10 50 100 500 1000 5000],...
				  'nStabSamp',0,...
				  'decodeFunc',decodeFunc,'inferFunc',inferFunc);

cd data/poliblog
if ~exist('srcfile','var')
	cd Processed;
	[examples,foldIdx] = loadPoliBlog(1);
	expSetup.foldIdx = foldIdx;
	cd ..;
else
	load(srcfile);
	expSetup.nFold = length(examples) / 4;
	expSetup.foldDist = [1 0 1 2];
end
cd ../..

if exist('filename','var')
	expSetup.save2file = filename;
end
% expSetup.optSGD = struct('maxIter',200);
experiment;

