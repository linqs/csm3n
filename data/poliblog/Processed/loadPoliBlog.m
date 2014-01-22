function examples = loadPoliBlog()
%
% Loads and formats the Political Blog data
%

examples = cell(16,1);
fprintf('Reading Feb1 ... ');
examples{1} = readExample('Feb_1');
fprintf('done.\n');
fprintf('Reading Feb2 ... ');
examples{2} = readExample('Feb_2');
fprintf('done.\n');
fprintf('Reading May1 ... ');
examples{3} = readExample('May_1');
fprintf('done.\n');
fprintf('Reading May2 ... ');
examples{4} = readExample('May_2');
fprintf('done.\n');

for shift = 1:3
	examples(4*shift+1:4*(shift+1)) = circshift(examples(1:4),shift);
end

