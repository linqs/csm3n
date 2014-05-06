function x = loadJpgOrBmp(prefix);

if exist(sprintf('%s.jpg', prefix), 'file') 
    x = imread(sprintf('%s.jpg', prefix));
elseif exist(sprintf('%s.JPG', prefix), 'file')
    x = imread(sprintf('%s.JPG', prefix));
elseif exist(sprintf('%s.bmp', prefix), 'file')
    x = imread(sprintf('%s.bmp', prefix));
else
    x = imread(sprintf('%s.BMP', prefix));
end