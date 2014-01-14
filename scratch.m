
clear;
examples = noisyX(6,1,0,1,0);
ex = examples{end};
x = reshape(squeeze(ex.Xnode(1,:,:)),[],1);

experiment;
w = w + 4*randn(size(w));

% decoder = @(nodePot,edgePot,edgeStruct) UGM_Decode_MaxOfMarginals(nodePot,edgePot,edgeStruct,@UGM_Infer_MeanField);
[f,g,x_p] = stabilityObj(w,ex,@UGM_Decode_LBP);

fprintf('Stability objective = %f\n', f);
fprintf('First 20 entries of x,x''\n');
disp([x(1:20) x_p(1:20)]);
fprintf('Num. permutations >= .50: %d\n', nnz(abs(x-x_p)>=.5));
fprintf('Num. permutations >= .25: %d\n', nnz(abs(x-x_p)>=.25));
fprintf('Num. permutations >= .10: %d\n', nnz(abs(x-x_p)>=.1));
fprintf('Num. permutations >= .01: %d\n', nnz(abs(x-x_p)>=.01));


[f,g] = csm3nObj(w,examples(1),examples(end),@UGM_Decode_LBP,100);
fprintf('CSM3N objective = %f\n', f);


clear;
examples = noisyX(10,1,0,1,0);
ex_l = examples(1:5);
ex_u = examples(6:10);
[w,fAvg] = trainCSM3N(ex_l,ex_u,@UGM_Decode_LBP,100)


clear;
examples = noisyX(4,1,0,1,0);
experiment;