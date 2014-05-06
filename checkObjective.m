function [] = checkObjective(func, x, epsilon)

d = length(x);
[~,dir] = func(x);
% dir = rand(d,1);
% dir(end) = 0;
dir = dir / norm(dir);

increments = linspace(-epsilon,epsilon,20);

y = zeros(size(increments));

for i = 1:length(increments)
    y(i) = func(x + increments(i) * dir);
end


plot(increments, y);

xlabel('distance from current point');
ylabel('objective value');

hold on;
plot(0, func(x), 'x');
hold off;

