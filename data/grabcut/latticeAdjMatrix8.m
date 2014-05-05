function [G, edgeDirections] = latticeAdjMatrix8(nRows, nCols)
%
% Creates a lattice graph with 8-neighborhood (i.e., N,S,E,W, NW, SW, NE, SE)
%
% Uses column-wise ordering of vertices, so
%   node at (i,j) is v = (j-1)*nRows + i = sub2ind([nRows nCols],i,j)
% To retrieve (i,j) from v, [i,j] = ind2sub([nRows nCols],v)

index = reshape(1:nRows*nCols, nRows, nCols);

% north-south
A = index(1:end-1,:);
B = index(2:end,:);
I = A(:);
J = B(:);
edgeD = ones(size(A(:)));

% east-west
A = index(:, 1:end-1);
B = index(:, 2:end);
I = [I; A(:)];
J = [J; B(:)];
edgeD = [edgeD; 2*ones(size(A(:)))];


% northwest-southeast
A = index(1:end-1, 1:end-1);
B = index(2:end, 2:end);
I = [I; A(:)];
J = [J; B(:)];
edgeD = [edgeD; 3*ones(size(A(:)))];


% northwest-southeast
A = index(2:end, 1:end-1);
B = index(1:end-1, 2:end);
I = [I; A(:)];
J = [J; B(:)];
edgeD = [edgeD; 4*ones(size(A(:)))];


G = sparse(I,J,true(length(I),1), numel(index), numel(index));
G = G + G';

edgeDirections = sparse(I,J,edgeD, numel(index), numel(index));
edgeDirections = edgeDirections + edgeDirections';