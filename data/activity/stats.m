
%% 5-action Dataset

clear

% dataset 1
% load file1.mat
% actions = 1:5;
% dataset 2
load file2.mat
actions = [1 2 3 5 6 7];

% constants
maxFrames = 1000;
maxBoxes = 100;

% global action transition matrix
trans = zeros(length(actions),length(actions));

% action co-occurrence matrix
cooccur_f = zeros(length(actions),length(actions));
cooccur_s = zeros(length(actions),length(actions));

% forall sequences
for s=1:length(anno)
	% list of actions in seq
	actions_s = zeros(length(actions));
	% forall frames
	for f=1:length(anno{s})
		frid = s*maxFrames + f;
		% list of actions in frame
		actions_f = zeros(length(actions));
		% forall bounding boxes
		for b=1:length(anno{s}{f})
			% bounding box ID
			bbid = frid*maxBoxes + b;
			% filter certain labels
			if any(actions == anno{s}{f}(b).act)
				% action ground-truth index
				a1 = find(actions == anno{s}{f}(b).act, 1, 'first');
				% update actions in sequence/frame
				actions_s(a1) = 1;
				actions_f(a1) = 1;
				% find bounding box in previous frame
				if f > 1
					for b_=1:length(anno{s}{f-1})
						bbid_ = (frid-1)*maxBoxes + b_;
						if anno{s}{f-1}(b_).id == anno{s}{f}(b).id
							a2 = find(actions == anno{s}{f-1}(b_).act, 1, 'first');
							trans(a1,a2) = trans(a1,a2) + 1;
						end
					end
				end
			end
		end
		% count action co-occurrences in frame
		for a1=1:length(actions)
			if actions_f(a1)
				for a2=a1:length(actions)
					if actions_f(a2)
						cooccur_f(a1,a2) = cooccur_f(a1,a2) + 1;
					end
				end
			end
		end
	end
	% count action co-occurrences in seq
	for a1=1:length(actions)
		if actions_s(a1)
			for a2=a1:length(actions)
				if actions_s(a2)
					cooccur_s(a1,a2) = cooccur_s(a1,a2) + 1;
				end
			end
		end
	end
end


%% OUTPUT ACTION TRANSITION MATRIX

% unnormalized
fprintf('Counts of action transitions\n\n')
disp(trans)

% normalized
Z = sum(trans,2);
Z(Z==0) = 1;
trans_norm = diag(Z.^-1) * trans;
fprintf('Normalized action transitions\n\n')
disp(trans_norm);


%% OUTPUT ACTION CO-OCCURRENCE MATRIX (IN FRAME)

% fill in lower triangle
cooccur_f = cooccur_f + triu(cooccur_f,1)';

% unnormalized
fprintf('Counts of action co-occurrence in frame\n\n')
disp(cooccur_f)

% normalized
Z = sum(cooccur_f,2);
Z(Z==0) = 1;
cooccur_f_norm = diag(Z.^-1) * cooccur_f;
fprintf('Normalized action co-occurrence in frame\n\n')
disp(cooccur_f_norm);


%% OUTPUT ACTION CO-OCCURRENCE MATRIX (IN SEQ)

% fill in lower triangle
cooccur_s = cooccur_s + triu(cooccur_s,1)';

% unnormalized
fprintf('Counts of action co-occurrence in sequence\n\n')
disp(cooccur_s)

% normalized
Z = sum(cooccur_s,2);
Z(Z==0) = 1;
cooccur_s_norm = diag(Z.^-1) * cooccur_s;
fprintf('Normalized action co-occurrence in sequence\n\n')
disp(cooccur_s_norm);

