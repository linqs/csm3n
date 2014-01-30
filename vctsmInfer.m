function y = vctsmInfer(w, kappa, Fx, Aeq, beq)
%
% Performs inference in a VCTSM model
%

% optimization options
clear options;
options.Method = 'lbfgs';
options.LS_type = 0;
options.LS_interp = 0;
% options.Display = 'off';

z = Fx' * w / kappa - 1;
lambda = zeros(size(beq));

fun = @(x, varargin) obj(x, z, kappa, Aeq, beq, varargin);
lambda = minFunc(fun, lambda, options, fun);

y = exp((Fx'*w + Aeq'*lambda)/kappa - 1);



% Subroutine to compute optimization objective.
function [f, g] = obj(lambda, z, kappa, Aeq, beq, varargin)
	y = exp(z + Aeq'*lambda / kappa);
	f = kappa * sum(y) - beq'*lambda;
	g = Aeq * y - beq;



