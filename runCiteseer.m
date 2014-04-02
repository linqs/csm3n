%% Citeseer experiment
%
% Variables:
%   nPC (def: 20)
%   runAlgos (def: [1 2 4])
%   inferFunc (def: UGM_Infer_TRBP)
%   decodeFunc (def: UGM_Decode_TRBP)
%   filename (def: will not save)

if ~exist('nPC','var')
	nPC = 20;
end
if ~exist('runAlgos','var')
	runAlgos = [1 2 4];
end
if ~exist('inferFunc','var')
	inferFunc = @UGM_Infer_TRBP;
end
if ~exist('decodeFunc','var')
	decodeFunc = @UGM_Decode_TRBP;
end

cd data;
[examples,foldIdx] = loadDocData('citeseer/citeseer.mat',4,nPC,1,1);
cd ..;
Xdesc = struct('discreteX',0,'nonneg',0);
expSetup = struct('Xdesc',Xdesc,...
				  'foldIdx',foldIdx,...
				  'runAlgos',runAlgos,...
				  'Cvec',[.001 .01 .1 .5 1 5 10 50 100 500 1000 5000],...
				  'nStabSamp',0,...
				  'decodeFunc',decodeFunc,'inferFunc',inferFunc);
if exist('filename','var')
	expSetup.save2file = filename;
end
% expSetup.optSGD = struct('maxIter',200);
experiment;

