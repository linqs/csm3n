function G = latticeAdjMatrix4(nRows, nCols)
%
% Creates a lattice graph with 4-neighborhood (i.e., N,S,E,W)
%
% Uses column-wise ordering of vertices, so
%   node at (i,j) is v = (j-1)*nRows + i = sub2ind([nRows nCols],i,j)
% To retrieve (i,j) from v, [i,j] = ind2sub([nRows nCols],v)

% Horizontal connections
diagVec1 = repmat([0; ones(nRows-1,1)],nCols,1); 

% Vertical connections
diagVec2 = ones(nRows*nCols,1);

% Make sparse matrix
G = spdiags([diagVec1 diagVec2],[1 nRows],nRows*nCols,nRows*nCols);
G = G + G';
