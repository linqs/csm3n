clear;
load grabCutProcessed;

maxDim = 100;

for i = 1:length(images)
    
    [m,n] = size(label{i});
    
    ratio = min(maxDim ./ [m n])
    
    newM = floor(ratio * m);
    newN = floor(ratio * n);
    
    images{i} = imresize(images{i}, [newM, newN]);
    %images{i}(images{i}>1) = 1.0;
    %images{i}(images{i}<0) = 0;
    
    subplot(211);
    imagesc(images{i});

    trimap{i} = imresize(trimap{i}, [newM, newN], 'nearest');
    label{i} = imresize(label{i}, [newM, newN], 'nearest');
    
    probBackgr{i} = imresize(reshape(probBackgr{i}, m, n), [newM, newN], 'nearest');
    probBackgr{i}(probBackgr{i}>1.0) = 1.0;
    probBackgr{i}(probBackgr{i}<0) = 0;

    probObject{i} = imresize(reshape(probObject{i}, m, n), [newM, newN], 'nearest');
    probObject{i}(probObject{i}>1.0) = 1.0;
    probObject{i}(probObject{i}<0) = 0;

    probObj{i} = imresize(probObj{i}, [newM, newN], 'nearest');
    probObj{i}(probObj{i}>1.0) = 1.0;
    probObj{i}(probObj{i}<0) = 0;
    
    subplot(321);
    imagesc(images{i});
    subplot(322);
    imagesc(trimap{i});
    subplot(323);
    imagesc(label{i});
    subplot(324);
    imagesc(probObject{i});
    subplot(325);
    imagesc(probBackgr{i});
    subplot(326);
    imagesc(probObj{i});
    drawnow;
end

save grabCutProcessedSmall;

