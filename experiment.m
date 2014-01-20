% Experimental testing harness
%
% Requires 1 variable, 1 optional:
% 	examples : cell array of (labeled) examples
% 	expSetup : (optional) structure containing experimental setup

assert(exist('examples','var') && iscell(examples), 'experiment requires cell array of examples.');
nEx = length(examples);
nParam = max(examples{1}.edgeMap(:));

% experiment vars
if ~exist('expSetup','var')
	expSetup = struct();
end
if isfield(expSetup,'nFold')
	nFold = expSetup.nFold;
else
	nFold = 1;
end
if isfield(expSetup,'foldDist')
	foldDist = expSetup.foldDist;
	nExFold = sum(foldDist);
	assert(nExFold <= nEx, 'Number of examples per fold greater than examples.');
	nTrain = foldDist(1);
	nUnlab = foldDist(2);
	nCV = foldDist(3);
	nTest = foldDist(4);
else
	nExFold = nEx / nFold;
	nTrain = 1;
	nUnlab = 1;
	nCV = 1;
	nTest = nExFold - 3;
end
if isfield(expSetup,'runAlgos')
	runAlgos = expSetup.runAlgos;
else
	runAlgos = 1:6;
end
if isfield(expSetup,'Cvec')
	Cvec = expSetup.Cvec;
else
	Cvec = 10.^linspace(-2,6,9);
end
if isfield(expSetup,'decoder')
	decoder = expSetup.decoder;
else
	decoder = @UGM_Decode_LBP;
end

% algorithm vars
algoNames = {'MLE', 'M3N', 'M3NLRR', 'VCTSM', 'CACC', 'CSM3N'};

% stability vars
% maxSamp = 10;
% nStabSamp = min(maxSamp, nNode*(nState-1));


%% MAIN LOOP

% job metadata
nJobs = length(runAlgos) * length(Cvec) * nFold;
totalTimer = tic;
count = 0;

% storage
params = cell(length(runAlgos), length(Cvec), nFold);
trErrs = zeros(length(runAlgos), length(Cvec), nFold);
cvErrs = zeros(length(runAlgos), length(Cvec), nFold);
teErrs = zeros(length(runAlgos), length(Cvec), nFold);

for fold = 1:nFold
	
	if (fold * nExFold) > nEx
		break;
	end
	
	fprintf('Starting fold %d of %d.\n', fold, nFold);
	
	% separate training/CV/testing
	fidx = (fold-1) * nExFold;
	tridx = fidx+1:fidx+nTrain;
	ulidx = fidx+nTrain+1:fidx+nTrain+nUnlab;
	cvidx = fidx+nTrain+nUnlab+1:fidx+nTrain+nUnlab+nCV;
	teidx = fidx+nTrain+nUnlab+nCV+1:fidx+nTrain+nUnlab+nCV++nTest;
	ex_tr = examples(tridx);
	ex_ul = examples(ulidx);
	ex_cv = examples(cvidx);
	ex_te = examples(teidx);
	
	
	for c = 1:length(Cvec)
		C = Cvec(c);
		
		for a = 1:length(runAlgos)

			%% TRAINING

			switch(runAlgos(a))
				
				% M(P)LE learning
				case 1
					fprintf('Training MLE ...\n');
					[w,nll] = trainMLE(ex_tr,@UGM_Infer_MeanField,C);
					params{a,c,fold}.w = w;

				% M3N learning
				case 2
					fprintf('Training M3N ...\n');
					[w,fAvg] = trainM3N(ex_tr,decoder,C);
					params{a,c,fold}.w = w;

				% M3NLRR learning (M3N with separate local/relational regularization)
				case 3
					fprintf('Training M3N with local/relational regularization ...\n');
					maxLocParamIdx = max(ex_tr{1}.nodeMap(:));
					relMultiplier = 100; % hack
					Csplit = C * ones(nParam,1);
					Csplit(maxLocParamIdx+1:end) = Csplit(maxLocParamIdx+1:end) * relMultiplier;
					[w,fAvg] = trainM3N(ex_tr,decoder,Csplit);
					params{a,c,fold}.w = w;

				% VCTSM learning (convexity optimization)
				case 4
					fprintf('Training VCTSM ...\n');
					[w,kappa,f] = trainVCTSM(ex_tr,C);
					params{a,c,fold}.w = w;
					params{a,c,fold}.kappa = kappa;

				% CACC learning (robust M3N)
				case 5
					fprintf('Training CACC ...\n');
					[w,fAvg] = trainCACC(ex_tr,decoder,C);
					params{a,c,fold}.w = w;
					
				% CSM3N learning (stability regularization)
				case 6
					fprintf('Training CSM3N ...\n');
					[w,fAvg] = trainCSM3N(ex_tr,ex_ul,decoder,0,.25);
					params{a,c,fold}.w = w;
					
			end
			
			% training stats
			errs = zeros(nTrain,1);
			for i = 1:nTrain
				if a ~= 4
					[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex_tr{i}.Xnode,ex_tr{i}.Xedge,ex_tr{i}.nodeMap,ex_tr{i}.edgeMap,ex_tr{i}.edgeStruct);
					pred = decoder(nodePot,edgePot,ex_tr{i}.edgeStruct);
				else
					mu = vctsmInfer(w,kappa,ex_tr{i}.Fx,ex_tr{i}.Aeq,ex_tr{i}.beq);
					pred = decodeMarginals(mu, ex_tr{i}.nNode, ex_tr{i}.nState);
				end
				errs(i) = nnz(ex_tr{i}.Y ~= pred) / ex_tr{i}.nNode;
			end
			trErrs(a,c,fold) = sum(errs)/nTrain;
			fprintf('Avg train err = %.4f\n', trErrs(a,c,fold));
			
			%% CROSS-VALIDATION
			
			errs = zeros(nCV,1);
			for i = 1:nCV
				if a ~= 4
					[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex_cv{i}.Xnode,ex_cv{i}.Xedge,ex_cv{i}.nodeMap,ex_cv{i}.edgeMap,ex_cv{i}.edgeStruct);
					pred = decoder(nodePot,edgePot,ex_cv{i}.edgeStruct);
				else
					mu = vctsmInfer(w,kappa,ex_cv{i}.Fx,ex_cv{i}.Aeq,ex_cv{i}.beq);
					pred = decodeMarginals(mu, ex_cv{i}.nNode, ex_cv{i}.nState);
				end
				errs(i) = nnz(ex_cv{i}.Y ~= pred()) / ex_cv{i}.nNode;
			end
			cvErrs(a,c,fold) = sum(errs)/nCV;
			fprintf('Avg CV err = %.4f\n', cvErrs(a,c,fold));
			
			%% TESTING
			
			errs = zeros(nTest,1);
			for i = 1:nTest
				if a ~= 4
					[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex_te{i}.Xnode,ex_te{i}.Xedge,ex_te{i}.nodeMap,ex_te{i}.edgeMap,ex_te{i}.edgeStruct);
					pred = decoder(nodePot,edgePot,ex_te{i}.edgeStruct);
				else
					mu = vctsmInfer(w,kappa,ex_te{i}.Fx,ex_te{i}.Aeq,ex_te{i}.beq);
					pred = decodeMarginals(mu, ex_te{i}.nNode, ex_te{i}.nState);
				end
				errs(i) = nnz(ex_te{i}.Y ~= pred()) / ex_te{i}.nNode;
				% plot prediction
				%subplot(length(runAlgos),1,a);
				%imagesc(reshape(pred,32,32));
				%title(algoNames(a));
			end
			teErrs(a,c,fold) = sum(errs)/nTest;
			fprintf('Avg test err = %.4f\n', teErrs(a,c,fold));
			
			%% PROGRESS
			
			count = count + 1;
			curTime = toc(totalTimer);
			fprintf('Finished %d of %d; elapsed: %.2f min; ETA: %.2f min\n', ...
				count, nJobs, curTime/60, (nJobs-count)*(curTime/count)/60);
			
		end

	end
	
	fprintf('\n');

end

% generalization error
geErrs = teErrs - trErrs;

% display results at end
colStr = {'Train','Valid','Test','Gen Err'};
for fold = 1:nFold
	disptable([trErrs(:,:,fold) cvErrs(:,:,fold) teErrs(:,:,fold) geErrs(:,:,fold)],colStr,algoNames,'%.5f');
end

