function examples = loadWeizmann1obj(makeEdgeDist,nEx)

if nargin < 1
	makeEdgeDist = 1;
end

cd 1obj;
load img_list.mat;

if nargin < 2 || nEx > length(fls)
	nEx = length(fls);
end

examples = cell(nEx,1);

for i = 1:nEx
	fname = fls(i).name;
	fprintf('Reading %s ... ',fname);
	stTime = tic;
	examples{i} = readExample(fname,makeEdgeDist);
	image(examples{i}.srcbw); drawnow;
	fprintf('done. Elapsed = %f sec.\n', toc(stTime));
end

cd ..;

