function examples = loadPoliBlog()
%
% Loads and formats the Political Blog data
%

examples = cell(4,1);
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


