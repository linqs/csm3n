clear;

%%
fin = fopen('fileList.txt','r');
[filenames] = textscan(fin, '%s\n');
filenames = filenames{1};
fclose(fin);

N = length(filenames);

for i = 1:N
    images{i} = double(loadJpgOrBmp(sprintf('data_GT/%s', filenames{i}))) / 255;
    trimap{i} = loadJpgOrBmp(sprintf('boundary_GT_lasso/%s', filenames{i}));
    label{i} = loadJpgOrBmp(sprintf('boundary_GT/%s', filenames{i}));

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

for i = 1:N
    
    vecImage = reshape(images{i}, numel(trimap{i}), 3);
    
    options = [];
    options.Display = 'final';
    options.MaxIter = 200;
    objectGM{i} = gmdistribution.fit(vecImage(trimap{i}==255,:), K, 'Regularize', 0.001, 'Options', options);
%     backgrGM{i} = gmdistribution.fit(vecImage(trimap{i}==0 | trimap{i}==64,:), K, 'Regularize', 0.01, 'Options', options);
     backgrGM{i} = gmdistribution.fit(vecImage(trimap{i}==64 | trimap{i}==0,:), K, 'Regularize', 0.001, 'Options', options);

    probObject = pdf(objectGM{i}, vecImage);
    probBackgr = pdf(backgrGM{i}, vecImage);
    
    probObj{i} = reshape(probObject ./ (probObject + probBackgr), size(label{i}));
    
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
    
end
save grabCutProcessed filenames images trimap label probObj objectGM backgrGM;

%%
clear;
load grabCutProcessed;

for i = 1:length(images)
    vecImage = reshape(images{i}, numel(trimap{i}), 3);
    
    probObject{i} = pdf(objectGM{i}, vecImage);
    probBackgr{i} = pdf(backgrGM{i}, vecImage); 
end
save grabCutProcessed probObject probBackgr filenames images trimap label probObj objectGM backgrGM;


%% shrink file size

clear;
load grabCutProcessed;

for i = 1:length(images)
    images{i} = uint8(images{i} * 255);
end

save grabCutProcessed probObject probBackgr filenames images trimap label probObj objectGM backgrGM;
