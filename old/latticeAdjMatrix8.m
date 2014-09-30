function G = latticeAdjMatrix8(nRows, nCols)

% Horizontal connections
diagVec1 = repmat([ones(nRows-1,1); 0],nCols,1);  
diagVec1 = diagVec1(1:end-1);

% Anti-diagonal connections
diagVec2 = [0; diagVec1(1:(nRows*(nCols-1)))];

% Vertical connections
diagVec3 = ones(nRows*(nCols-1),1);

% Diagonal connections
diagVec4 = diagVec2(2:end-1);

% Sum diagonals
G = diag(diagVec1,1) + ...
    diag(diagVec2,nRows-1) + ...
    diag(diagVec3,nRows) + ...
    diag(diagVec4,nRows+1);
G = G + G';
