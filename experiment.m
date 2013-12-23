
% Experimental testing harness

clear
load crf_samples.mat

% experiment vars
[nNodeY,nEx] = size(Y);
nFold = 10;
nExFold = nEx / nFold;
nTrain = nExFold - 2;
nCV = 1;
nTest = 1;

% algorithm vars
algoNames = {'MLE', 'M3N', 'CSM3N'};
algoTypes = [1];

% crossvalidation vars
Cvec = nNodeY;%10.^linspace(-2,6,9);

% stability vars
% maxSamp = 10;
% nStabSamp = min(maxSamp, nNode*(nState-1));

% make edge structure for Y vars only
edgeStruct = UGM_makeEdgeStruct(G,nStateY,0);


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
	Xtr = X(tridx,:,:);
	Ycv = Y(:,cvidx);
	Xcv = X(cvidx,:,:);
	Yte = Y(:,teidx);
	Xte = X(teidx,:,:);
	

	%% TRAINING / CROSS-VALIDATION
	
	for c = 1:length(Cvec)
		C = Cvec(c);
		
		for a = 1:length(algoTypes)

			% training
			switch(algoTypes(a))

				% MLE learning
				case 1
					[w,nll] = trainMLE(Xtr,Ytr,edgeStruct,@UGM_Infer_MeanField,C)

				% M3N learning
				case 2
					[w,loss] = trainM3N(Xtr,Ytr,edgeStruct,@UGM_Decode_LinProg,C,1)

				% CSM3N learning
				case 3

			end
			
			% CV stats
			Xedge = UGM_makeEdgeFeatures(Xcv,edgeStruct.edgeEnds);
			[nodeMap,edgeMap] = UGM_makeCRFmaps(Xcv,Xedge,edgeStruct,0,1);
			for i = 1:nCV
				% infer (decode) labels of CV example
				[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xcv,Xedge,nodeMap,edgeMap,edgeStruct,i);
				edgeStruct.useMex = 0; % mex decoding doesn't work
				pred = UGM_Decode_LinProg(nodePot,edgePot,edgeStruct);
				errs = nnz(Ycv(:,i) ~= pred(1:nNodeY))
				edgeStruct.useMex = 1;
			end
			
		end

	end



	%% TESTING


end
