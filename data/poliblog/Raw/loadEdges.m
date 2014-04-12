fin = fopen('Edges.edge', 'r');

L = [];
while ~feof(fin)
    line = strtrim(fgetl(fin));
    if strcmp(line, 'edge [')
        srcLine = strtrim(fgetl(fin));
        tarLine = strtrim(fgetl(fin));
        
        [~, src] = strtok(srcLine);
        [~, tar] = strtok(tarLine);
        
        L = [L; str2double(src), str2double(tar)];
    end
end

fclose(fin);