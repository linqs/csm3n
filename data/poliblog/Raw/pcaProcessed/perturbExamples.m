function perturbed = perturbExamples(examples,eta)

perturbed = cell(length(examples),1);
for i = 1:length(examples)
	ex = examples{i};
	for n = 1:ex.nNode
		for f = 1:ex.nNodeFeat
			if rand() < eta
				ex.Xnode(1,f,n) = ~ex.Xnode(1,f,n);
			end
		end
	end
	ex.Xedge = makeEdgeFeatures(ex.Xnode,ex.edgeStruct.edgeEnds);
	perturbed{i} = ex;
end
