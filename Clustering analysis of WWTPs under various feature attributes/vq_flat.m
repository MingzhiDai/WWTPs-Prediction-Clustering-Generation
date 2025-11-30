function [cost,cm,cv,cc,cs,map] = vq_flat( vec, nc, cvtype, mincs, maxit)
%
% vq_flat -- greedy vector quantization learning
%
% Usage:
%   [cost,cm,cv,cc,cs,map] = vq_flat( vec, nc, cvtype, mincs, maxit)
% Input:
%   vec		list of column vectors
%   nc		number of clusters
%   cvtype	covariance type:
%		0 - unit covariance
%		1 - diagonal covariance (default)
%		2 - inverse of diagonal covariance
%		3 - full covariance
%   mincs	minimal cluster size (default 1)
%   maxit	maximum number of iterations (default 100)
% Output:
%   cost	quantization cost
%   cm		cluster means
%   cv		cluster variances
%   cc		cluster costs
%   cs		cluster sizes
%   map		vector to cluster mapping
%

