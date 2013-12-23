
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
algoTypes = [1 2];%[1 2 3];

% crossvalidation vars
Cvec = nNodeY;%10.^linspace(-2,6,9);

% stability vars
% maxSamp = 10;
% nStabSamp = min(maxSamp, nNode*(nState-1));

% make edge structure for Y vars only
edgeStruct = UGM_makeEdgeStruct(G,nStateY,1);


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
			%fprintf('Starting learning.\n')
			Xedge = UGM_makeEdgeFeatures(Xtr,edgeStruct.edgeEnds);
			[nodeMap,edgeMap,w] = UGM_makeCRFmaps(Xtr,Xedge,edgeStruct,0,1);
			edgeStruct.useMex = 0; % none of the methods seem to work with mex
			switch(algoTypes(a))
				
				% random weights (for bypassing learning)
				case 0
					w = rand(size(w));

				% M(P)LE learning
				case 1
					%[w,nll] = trainMLE(Xtr,Ytr,edgeStruct,@UGM_Infer_MeanField,C)
					[w,nll] = trainMLE(Xtr,Ytr,edgeStruct,0,C)

				% M3N learning
				case 2
					[w,loss] = trainM3N(Xtr,Ytr,edgeStruct,@UGM_Decode_LinProg,C,1)

				% CSM3N learning
				case 3

			end
			edgeStruct.useMex = 1; % turn mex back on
			
			% training stats
			trainErrs = zeros(nTrain,1);
			for i = 1:nTrain
				% infer (decode) labels of CV example
				[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xtr,Xedge,nodeMap,edgeMap,edgeStruct,i);
				%pred = UGM_Decode_LinProg(nodePot,edgePot,edgeStruct);
				pred = UGM_Decode_MaxOfMarginals(nodePot,edgePot,edgeStruct,@UGM_Infer_LBP);
				trainErrs(i) = ( nnz(Ytr(:,i) ~= pred(1:nNodeY)) ) / nNodeY;
			end
			fprintf('Avg train err = %.4f\n', sum(trainErrs)/nTrain);
			
			% CV stats
			%fprintf('Starting cross-validation.\n')
			Xedge = UGM_makeEdgeFeatures(Xcv,edgeStruct.edgeEnds);
			[nodeMap,edgeMap] = UGM_makeCRFmaps(Xcv,Xedge,edgeStruct,0,1);
			cvErrs = zeros(nCV,1);
			for i = 1:nCV
				% infer (decode) labels of CV example
				[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xcv,Xedge,nodeMap,edgeMap,edgeStruct,i);
				%pred = UGM_Decode_LinProg(nodePot,edgePot,edgeStruct);
				pred = UGM_Decode_MaxOfMarginals(nodePot,edgePot,edgeStruct,@UGM_Infer_LBP);
				cvErrs(i) = ( nnz(Ycv(:,i) ~= pred(1:nNodeY)) ) / nNodeY;
			end
			fprintf('Avg CV err = %.4f\n', sum(cvErrs)/nCV);
			
		end

	end



	%% TESTING


end
