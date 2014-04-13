
clear;

crawls = {'02092012', '05092102'};

figure(1);
for i = 1:length(crawls)
    
     % load from text file if .mat file hasn't been saved
     % if the text file has been updated, the mat file MUST be removed
     if exist(sprintf('wordIJ_%s.mat', crawls{i}), 'file')
         load(sprintf('wordIJ_%s.mat', crawls{i}));
     else
        IJ = load(sprintf('wordIJ_%s.txt', crawls{i}));
        save(sprintf('wordIJ_%s.mat', crawls{i}), 'IJ');
     end
    
    X{i} = sparse(IJ(:,1), IJ(:,2), ones(length(IJ),1));
    X{i}(1490,:) = 0; % add the last missing page for indexing convenience
    subplot(2,1,i);
    spy(X{i});
    axis normal;
    title(crawls{i});
end

% fix array sizes if new words were added in second crawl
if size(X{1},2) < size(X{2}, 2)
    X{1}(:,size(X{2},2)) = 0;
end

%% load word list

[wordIndex, wordStrings] = textread('wordIndex.txt', '%d %s\n');

% reorder according to index
wordStrings(wordIndex) = wordStrings;

%% combine both crawls

X = X{1} + X{2} > 0;


%% prune overly common or rare words

N = size(X,1);

DF = full(sum(X>0,1)) / N;


maxDF = 0.25;
minDF = 50 / N;

keptWords = DF <= maxDF & DF >= minDF;

fprintf('Number of unpruned words: %d\n', nnz(keptWords));

Xpruned = double(X(:, keptWords) > 0);

DFpruned = DF(keptWords);

%% remove empty pages

remove = sum(Xpruned,2)==0;

Xpruned = Xpruned(~remove,:);

N = size(Xpruned,1);

%% normalize X to unit vectors

% 
% Xpruned = bsxfun(@rdivide, Xpruned, sqrt(sum(Xpruned.*Xpruned,2)));


%% output dictionary and pruned words
commonWords = wordStrings(DF > maxDF);
rareWords = wordStrings(DF < minDF);
filteredWords = wordStrings(keptWords);

fout = fopen('prunedCommonWords.txt', 'w');
for i = 1:length(commonWords)
    fprintf(fout, '%s\n', commonWords{i});
end
fclose(fout);

fout = fopen('prunedRareWords.txt', 'w');
for i = 1:length(rareWords)
    fprintf(fout, '%s\n', rareWords{i});
end
fclose(fout);

fout = fopen('filteredWords.txt', 'w');
for i = 1:length(filteredWords)
    fprintf(fout, '%s\n', filteredWords{i});
end
fclose(fout);

%% PCA dimensionality reduction

figure(2);

d = 50; % target dimensionality

% center data
means = mean(full(Xpruned));
Xcentered = bsxfun(@minus, full(Xpruned), means);

% compute covariance
cov = Xcentered'*Xcentered;

% compute eigendecomposition and plot spectrum

[V,D] = eigs(cov, d);

subplot(411)
stem(diag(D));
title('eigenspectrum');


%% compare quality of low-rank reconstruction

subplot(412);

imagesc(cov);
title('original covariance matrix');

subplot(413);
imagesc(V(:,1:d)*D(1:d,1:d)*V(:,1:d)');
title('low-rank reconstruction');

%% project data onto principle components

scaling = ones(1,d);%diag(D(1:d,1:d))';

PCs = bsxfun(@rdivide, V(:,1:d), scaling);

Xpc = Xcentered * PCs;

subplot(427);

imagesc(Xpc);
title('Low-Dim X');


Xrecon = bsxfun(@plus, bsxfun(@times, scaling, Xpc) * PCs', means);

subplot(428)
imagesc(Xrecon);
title('Reconstructed X');

fprintf('low-rank reconstruction error: %f\n', norm(Xrecon(:) - Xpruned(:)));

%% load links and labels

load('L.mat');
Y = load(sprintf('%s/Y.csv', crawls{1}));

Y = Y(~remove);

links = full(sparse(L(:,1), L(:,2), ones(size(L,1), 1)));

links = links(~remove, ~remove);


%% split data and output files


fold = mod(1:N, 4) + 1;
fold = fold(randperm(N));

for i = 1:4
    [I,J] = find(links(fold==i, fold==i));
    
    examples{i}.link = [I J];
    
    examples{i}.Y = Y(fold==i);
    examples{i}.X = Xpc(fold==i,:);
    
    csvwrite(sprintf('pcaProcessed/split%d.Link.csv', i), examples{i}.link);
    csvwrite(sprintf('pcaProcessed/split%d.Word.csv', i), examples{i}.X);
    csvwrite(sprintf('pcaProcessed/split%d.Y.csv', i), examples{i}.Y);
end



%%

figure(3);

subplot(311);
plot(Y);
title('Label');

subplot(312);
Kfull = full(double(X)*double(X)');
imagesc(Kfull./bsxfun(@times, sqrt(diag(Kfull)), sqrt(diag(Kfull)')));
title('Normalized full kernel');

subplot(313);
K = Xpc*Xpc';
imagesc(K./bsxfun(@times, sqrt(diag(K)), sqrt(diag(K)')));
title('normalized PCA reconstructed kernel');


