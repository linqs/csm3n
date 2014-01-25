function examples = loadWeizmann1obj(makeEdgeDist)

if nargin < 1
	makeEdgeDist = 0;
end

cd 1obj;
load img_list.mat;

nEx = length(fls);

examples = cell(nEx,1);

for i = 1:nEx
	fname = fls(i).name;
	examples{i} = readExample(fname,makeEdgeDist);
end

cd ..;

