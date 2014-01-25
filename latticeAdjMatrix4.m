function G = latticeAdjMatrix4(nRows, nCols)

% Horizontal connections
diagVec1 = repmat([0; ones(nRows-1,1)],nCols,1); 

% Vertical connections
diagVec2 = ones(nRows*nCols,1);

% Make sparese matrix
G = spdiags([diagVec1 diagVec2],[1 nRows],nRows*nCols,nRows*nCols);
G = G + G';
