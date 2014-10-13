function plotPredictions(examples,expSetup,fold,params,bestParam,vctsmIdx)

teidx = expSetup.foldIdx(fold).teidx;
ex_te = examples(teidx);

w = params{vctsmIdx,fold,bestParam(vctsmIdx,fold)}.w;
kappa = params{vctsmIdx,fold,bestParam(vctsmIdx,fold)}.kappa;

for i = 1:10
		
	ex = ex_te{i};
	figure(1);
% 	subplot(1,2,1);
	noisyimg = zeros(ex.nNode,1);
	nFeat = size(ex.Xnode,2);
	for n = 1:ex.nNode
		noisyimg(n) = find(ex.Xnode(1,:,n)) / nFeat;
	end
	imagesc(reshape(noisyimg,42,60)); colormap gray; axis off; set(gca,'Position',[0 0 1 1]);
	
	[nodePot,edgePot] = UGM_CRF_makePotentials(w,ex.Xnode,ex.Xedge,ex.nodeMap,ex.edgeMap,ex.edgeStruct);
	pred = UGM_Decode_ConvexBP(kappa,nodePot,edgePot,ex.edgeStruct,expSetup.inferFunc);
	figure(2);
% 	subplot(1,2,2);
	imagesc(reshape(pred,42,60)); colormap gray; axis off; set(gca,'Position',[0 0 1 1]);

	pause
end
