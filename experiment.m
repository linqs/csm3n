% Experimental testing harness

% experiment vars
[nNodeY,nEx] = size(Y);
nFold = 1;%nSamp;
nExFold = nEx / nFold;
nTrain = 1;
nCV = 1;
nTest = nExFold - nTrain - nCV;

% algorithm vars
algoNames = {'MLE', 'M3N', 'CSM3N'};
algoTypes = [1];

% crossvalidation vars
Cvec = nNodeY;%10.^linspace(-2,6,9);

% stability vars
% maxSamp = 10;
% nStabSamp = min(maxSamp, nNode*(nState-1));

% make node/edge maps for UGM (might have to move to main loop)
[nodeMap,edgeMap,w] = makeRelMRFmaps(edgeStruct,edgeType);


%% MAIN LOOP
nJobs = length(algoTypes) * length(Cvec) * nFold;
totalTimer = tic;
count = 0;
for fold = 1:nFold
	
	% separate training/CV/testing
	fidx = (fold-1) * nExFold;
	tridx = fidx+1:fidx+nTrain;
	cvidx = fidx+nTrain+1:fidx+nTrain+nCV;
	teidx = fidx+nTrain+nCV+1:fidx+nExFold;
	Ytr = Y(:,tridx);
	Xtr = X(:,tridx);
	Ycv = Y(:,cvidx);
	Xcv = X(:,cvidx);
	Yte = Y(:,teidx);
	Xte = X(:,teidx);
	

	%% TRAINING / CROSS-VALIDATION
	
	for c = 1:length(Cvec)
		C = Cvec(c);
		
		for a = 1:algoTypes

			% training
			switch(a)

				% MLE learning
				case 1
					[w,nll] = trainMLE(Xtr,Ytr,edgeStruct,nodeMap,edgeMap,@UGM_Infer_MeanField,C)
					%break;

				% M3N learning
				case 2
					break;

				% CSM3N learning
				case 3
					break;

			end
			
			% CV stats
			[nodePot,edgePot] = UGM_MRF_makePotentials(w,nodeMap,edgeMap,edgeStruct);
			for i = 1:nCV
				% infer (decode) labels of CV example
				clamped = [zeros(size(Ycv(:,i))) ; Xcv(:,i)];
				edgeStruct.useMex = 0; % for now, don't use mex
				pred = UGM_Decode_Conditional(nodePot,edgePot,edgeStruct,clamped,@UGM_Decode_LBP);
			end
			
		end

	end



	%% TESTING


end
