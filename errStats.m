function [cm,err,pre,rec,f1,f1avg,f1wavg] = errStats(truth,pred)

cm = confusionmat(truth,pred);
nLabels = size(cm,1);

err = sum(diag(cm)) / sum(cm(:));

f1avg = 0;
f1wavg = 0;
for l = 1:nLabels
	if sum(cm(:,l)) == 0
		pre(l) = 1;
	else
		pre(l) = cm(l,l) / sum(cm(:,l));
	end
	if sum(cm(l,:)) == 0
		rec(l) = 1;
	else
		rec(l) = cm(l,l) / sum(cm(l,:));
	end
	if pre(l) ~= 0 || rec(l) ~= 0
		f1(l) = 2*pre(l)*rec(l) / (pre(l)+rec(l));
		f1avg = f1avg + f1(l) * (1/nLabels);
		f1wavg = f1wavg + f1(l) * (sum(cm(l,:))/sum(cm(:)));
	end
end

