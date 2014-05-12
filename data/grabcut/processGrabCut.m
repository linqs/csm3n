clear;

maxDim = 100;

%%
fin = fopen('fileList.txt','r');
[filenames] = textscan(fin, '%s\n');
filenames = filenames{1};
fclose(fin);

N = length(filenames);

for i = 1:N
    images{i} = loadJpgOrBmp(sprintf('data_GT/%s', filenames{i}));
    trimap{i} = loadJpgOrBmp(sprintf('boundary_GT_lasso/%s', filenames{i}));
    label{i} = loadJpgOrBmp(sprintf('boundary_GT/%s', filenames{i}));
    
    % shrink images
    [m,n] = size(label{i});
    ratio = min(maxDim ./ [m n]);
    
    newM = floor(ratio * m);
    newN = floor(ratio * n);
    
    images{i} = imresize(images{i}, [newM, newN]);
    trimap{i} = imresize(trimap{i}, [newM, newN], 'nearest');
    label{i} = imresize(label{i}, [newM, newN], 'nearest');

    subplot(311);
    imagesc(images{i});
    subplot(312);
    imagesc(trimap{i});
    subplot(313);
    imagesc(label{i});
    drawnow;
end


%% fit gaussians

K = 30;

timer = tic;

for i = 1:N
    
    vecImage = reshape(double(images{i}), numel(trimap{i}), 3);
    
    options = [];
    options.Display = 'final';
    options.MaxIter = 500;
    objectGM{i} = gmdistribution.fit(vecImage(trimap{i}==255,:), K, 'Regularize', 0.001, 'Options', options);
    backgrGM{i} = gmdistribution.fit(vecImage(trimap{i}==64,:), K, 'Regularize', 0.001, 'Options', options);

    probObject{i} = reshape(pdf(objectGM{i}, vecImage), size(label{i}));
    probBackgr{i} = reshape(pdf(backgrGM{i}, vecImage), size(label{i}));
    
    probObj{i} = probObject{i} ./ (probObject{i} + probBackgr{i});
    
    subplot(311);
    imagesc(images{i});
    subplot(312);
    imagesc(trimap{i});
    subplot(313);
    title('Trimap');
    imagesc(probObj{i});
    title('Mixture of Gaussian Feature');
    colormap gray;
    drawnow;
    
    fprintf('Finished %d of %d in %2.2f minutes. Eta %2.2f minutes\n', i, N, toc(timer)/60, (toc(timer)/i) * (N-i)/60);
    
end
save grabCutProcessed filenames images trimap label probObj objectGM backgrGM probObject probBackgr;

