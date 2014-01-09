function y = relaxedLossAugInfer(wtFx, yhat, Aeq, beq, nLoc)


ell = zeros(size(yhat));
ell(1:nLoc) = 1 - 2*yhat(1:nLoc);
A = -Aeq';
b = wtFx' + ell;
Ehat = wtFx * yhat;

obj = @(x) lambdaObj(x,Ehat,beq);
[x, f] = fmincon(obj,x0,A,b);


function f = lambdaObj(x, Ehat, beq)

	f = -Ehat - x'*beq;
	
