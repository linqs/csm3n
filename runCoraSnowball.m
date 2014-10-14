%% Cora experiment
%
% Variables:
%   nFold (def: 10)
%   nPC (def: 100)
%   runAlgos (def: [4 7])
%   inferFunc (def: UGM_Infer_CountBP)
%   decodeFunc (def: UGM_Decode_LBP)
%	Cvec (def: [.0001 .0005 .001 .0025 .005 .0075 .01 .025 .05 .1])
%   save2file (def: will not save)

if ~exist('nFold','var')
	nFold = 10;
end
if ~exist('nPC','var')
	nPC = 100;
end
if ~exist('runAlgos','var')
	runAlgos = [4 7];
end
if ~exist('inferFunc','var')
	inferFunc = @UGM_Infer_CountBP;
end
if ~exist('decodeFunc','var')
	decodeFunc = @UGM_Decode_LBP;
end
if ~exist('Cvec','var')
% 	Cvec = [.0005 .001 .0025 .005 .0075 .01 .025 .05 .1];
	Cvec = [.00001 .00005 .0001 .0005 .001 .005 .01 .05 .1];
end

if any(runAlgos == 2) || any(runAlgos == 3)
	stepSizeVec = [.1 .2 .5 1 2];
else
	stepSizeVec = 1;
end

if any(runAlgos == 5)
	kappaVec = [.1 .2 .5 1 2 5 10];
else
	kappaVec = 1;
end

% seed the RNG
rng(917);

cd data;
examples = {};
for f = 1:nFold
	examples(end+1:end+3) = loadDocDataSnowball('cora/cora.mat',3,nPC,0.1,[],1,2);
	foldIdx(f).tridx = 3*(f-1)+1;
	foldIdx(f).ulidx = [];
	foldIdx(f).cvidx = 3*(f-1)+2;
	foldIdx(f).teidx = 3*(f-1)+3;
end
cd ..;

maxIter = 200;

expSetup = struct('foldIdx',foldIdx ...
				 ,'runAlgos',runAlgos ...
				 ,'decodeFunc',decodeFunc,'inferFunc',inferFunc ...
				 ,'Cvec',Cvec ...
				 ,'stepSizeVec',stepSizeVec ...
				 ,'kappaVec',kappaVec ...
				 );
expSetup.optSGD = struct('maxIter',maxIter ...
						,'plotObj',0,'plotRefresh',10 ...
						,'verbose',0,'returnBest',1);
expSetup.optLBFGS = struct('Display','iter','verbose',0 ...
						  ,'plotObj',0,'plotRefresh',10 ...
						  ,'MaxIter',maxIter,'MaxFunEvals',maxIter);
			  
if exist('save2file','var')
	expSetup.save2file = save2file;
end

fprintf('\nSTARTING EXPERIMENT\n\n')

experiment;

