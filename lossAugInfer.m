function y = lossAugInfer(w, Xnode, Xedge, Ynode, nodeMap, edgeMap, edgeStruct, decodeFunc, varargin)
%
% Performs loss-augmented inference, using the supplied decoder function.
% 
% Xnode : nNodeFeat x nNode, X node values
% Xedge : nEdgeFeat x nEdge, X edge features
% Ynode : nState x nNode, Y node values
% nodeMap : nNode x nState x nFeat, maps nodes to parameters
% edgeMap : nState x nState x nEdge x 2*nFeat, maps edges to parameters
% edgeStruct: edge structure
% decodeFunc : decoder function

% parse input
assert(nargin >= 8, 'USAGE: lossAugInfer(w,Xnode,Xedge,Ynode,nodeMap,edgeMap,edgeStruct,decodeFunc)')

% make potentials
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);

% loss augmentation
nodePot = nodePot .* exp(1 - Ynode');

% loss-augmented decoding
y = decodeFunc(nodePot,edgePot,edgeStruct,varargin{:});

