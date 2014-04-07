function [cm,err,pre,rec,f1w,f1avg] = errStats(truth,pred)

cm = confusionmat(truth,pred);

err = sum(diag(cm)) / sum(cm(:));

f1w = 0;
f1avg = 0;
for l = 1:size(cm,1)
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
		f1w = f1w + (sum(cm(l,:))/sum(cm(:))) * (2*pre(l)*rec(l)/(pre(l)+rec(l)));
		f1avg = f1avg + (1/size(cm,1)) * (2*pre(l)*rec(l)/(pre(l)+rec(l)));
	end
end

