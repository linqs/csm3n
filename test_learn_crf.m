% Tests learning parameters of CRF.

load crf_samples.mat
Y = samples(1:nNode,:);
X = samples(nNode+1:end,:);

% experiment variables
nFold = 10;
nExFold = size(Y,2) / nFold;
nTrain = 1;
nTest = nExFold - nTrain;


%% MAIN LOOP
for fold=1:nFold
	

	%% TRAINING



	%% TESTING


end
