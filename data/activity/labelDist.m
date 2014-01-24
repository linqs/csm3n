function dist = labelDist(anno,actions)

dist = zeros(length(anno),length(actions));
for s = 1:length(anno)
	for f = 1:length(anno{s})
		for b = 1:length(anno{s}{f})
			a = find(actions == anno{s}{f}(b).act);
			if isempty(a)
				continue;
			end
			dist(s,a) = dist(s,a) + 1;
		end
	end
end
