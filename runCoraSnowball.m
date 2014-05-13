%% Cora experiment
%
% Variables:
%   nFold (def: 10)
%   nPC (def: 20)
%   runAlgos (def: [4 10])
%   inferFunc (def: UGM_Infer_TRBP)
%   decodeFunc (def: UGM_Decode_TRBP)
%   save2file (def: will not save)

if ~exist('nFold','var')
	nFold = 10;
end
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
examples = {};
for f = 1:nFold
	examples(end+1:end+3) = loadDocDataSnowball('cora/cora.mat',3,nPC,0.01,[],1,0);
	foldIdx(f).tridx = 3*(f-1)+1;
	foldIdx(f).ulidx = [];
	foldIdx(f).cvidx = 3*(f-1)+2;
	foldIdx(f).teidx = 3*(f-1)+3;
end
cd ..;

maxIter = 200;

expSetup = struct('foldIdx',foldIdx ...
				 ,'runAlgos',runAlgos ...
				 ,'decodeFunc',@UGM_Decode_TRBP,'inferFunc',@UGM_Infer_TRBP ...
				 ,'Cvec',[.001 .01 .1 1 10] ...
				 ,'Cvec2',[.001 .01 .1 1 10] ...
				 ,'stepSizeVec',1 ...%[.1 .2 .5 1 2 5 10] ...
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

experiment;

