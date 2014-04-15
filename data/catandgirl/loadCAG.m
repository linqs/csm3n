function [examples] = loadCAG(noiseRate)

% Load all GIF files in dir
filelist = dir(pwd);
i = 0;
for f = 1:length(filelist)
	if length(filelist(f).name) < 4 || ~strcmp(filelist(f).name(end-3:end),'.gif')
		continue
	end
	% Load GIF
	i = i + 1;
	examples{i} = loadExample(filelist(f).name,noiseRate,1,0,0);
end
