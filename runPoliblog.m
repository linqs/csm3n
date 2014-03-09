%% Poliblog experiment
%
% Variables:
%   runAlgos (def: [1 2 4])
%   inferFunc (def: UGM_Infer_LBP)
%   decodeFunc (def: UGM_Decode_LBP)
%   filename (def: will not save)

if ~exist('runAlgos','var')
	runAlgos = [1 2 4];
end
if ~exist('inferFunc','var')
	inferFunc = @UGM_Infer_LBP;
end
if ~exist('decodeFunc','var')
	decodeFunc = @UGM_Decode_LBP;
end

cd data/poliblog/Processed;
[examples,foldIdx] = loadPoliBlog(1);
cd ../../..;
Xdesc = struct('discreteX',1,'nonneg',1);
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

