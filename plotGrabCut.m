function plotGrabCut(pred, ex, expSetup)

if isfield(expSetup, 'plotFuncAxis')
    ax = expSetup.plotFuncAxis;
else
    ax = gca;
end

buffer = 0.75*ones(10, size(ex.srcbw,2));

img = [double(ex.srcrgb(:,:,1)); buffer; ex.probObject; buffer; ex.probBackgr];

tmp = 0.5*ones(size(ex.srcbw));
tmp(ex.mask) = ex.probObject(ex.mask) < ex.probBackgr(ex.mask);
img = [img; buffer; tmp];


tmp = 0.5*ones(size(ex.srcbw));
tmp(ex.mask) = pred - 1;
img = [img; buffer; tmp];

% make rgb
rgb = zeros(size(img, 1), size(img, 2), 3);
for i = 1:3
    img(1:size(ex.srcbw,1), :) = double(ex.srcrgb(:,:,i))/255;
    rgb(:,:,i) = img;
end

imagesc(rgb, 'Parent', ax);
colormap gray;
axis image;

ylabel('Prediction, Local Threshold, Background Prob, Foreground Prob, Original');
drawnow;
