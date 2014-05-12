function [] = testFoldIdx(foldIdx)

nFold = length(foldIdx);

img = zeros(nFold, 0, 3);

for i = 1:length(foldIdx)
    img(i, foldIdx(i).tridx, 1) = 1.0;
    img(i, foldIdx(i).cvidx, 2) = 1.0;
    img(i, foldIdx(i).teidx, 3) = 1.0;
    
    assert(isempty(intersect(foldIdx(i).tridx, foldIdx(i).cvidx)), 'Train and validation sets overlap');
    assert(isempty(intersect(foldIdx(i).tridx, foldIdx(i).teidx)), 'Train and test sets overlap');
    assert(isempty(intersect(foldIdx(i).cvidx, foldIdx(i).teidx)), 'Validation and test sets overlap');
end

imagesc(img);
ylabel('Fold');
xlabel('Example index');
title('Red: train, green: validation, blue: test');
