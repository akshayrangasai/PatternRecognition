load 'a.train.txt'
load 'bA.train.txt'
load 'chA.train.txt'
load 'dA.train.txt'
load 'lA.train.txt'
load 'LA.train.txt'
load 'tA.train.txt'
load 'ai.train.txt'

[~, C] = kmeans([a_train; a_train; ai_train; chA_train; dA_train; lA_train; LA_train; tA_train;], 20);

files1 = dir('a.train/*.txt');
files2 = dir('ai.train/*.txt');
files3 = dir('bA.train/*.txt');
files4 = dir('chA.train/*.txt');
files5 = dir('dA.train/*.txt');
files6 = dir('lA.train/*.txt');
files7 = dir('LA.train/*.txt');
files8 = dir('tA.train/*.txt');

cd a.train/
mkdir hmm_files
for file = files1'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_files/',file.name), states, ' ');
end
system('cat hmm_files/*.txt > ../A.TRAIN.HMM.SEQ');
cd ..

cd ai.train/
mkdir hmm_files
for file = files2'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_files/',file.name), states, ' ');
end
system('cat hmm_files/*.txt > ../AI.TRAIN.HMM.SEQ');
cd ..

cd bA.train/
mkdir hmm_files
for file = files3'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_files/',file.name), states, ' ');
end
system('cat hmm_files/*.txt > ../BA.TRAIN.HMM.SEQ');
cd ..

cd chA.train/
mkdir hmm_files
for file = files4'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_files/',file.name), states, ' ');
end
system('cat hmm_files/*.txt > ../CHA.TRAIN.HMM.SEQ');
cd ..

cd dA.train/
mkdir hmm_files
for file = files5'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_files/',file.name), states, ' ');
end
system('cat hmm_files/*.txt > ../DA.TRAIN.HMM.SEQ');
cd ..

cd lA.train/
mkdir hmm_files
for file = files6'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_files/',file.name), states, ' ');
end
system('cat hmm_files/*.txt > ../LA0.TRAIN.HMM.SEQ');
cd ..

cd LA.train/
mkdir hmm_files
for file = files7'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_files/',file.name), states, ' ');
end
system('cat hmm_files/*.txt > ../LA1.TRAIN.HMM.SEQ');
cd ..

cd tA.train/
mkdir hmm_files
for file = files8'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_files/',file.name), states, ' ');
end
system('cat hmm_files/*.txt > ../TA.TRAIN.HMM.SEQ');
cd ..


%--------------------
% cd eight/test/
% mkdir hmm_eight
% for file = files8'
%     tmp = load(file.name);
%     states = [];
%     for i = 1:length(tmp)
%         ss = (C - repmat(tmp(i,:), 20, 1)).^2;
%         d = sum(ss,2);
%         [~,I] = min(d);
%         states = [states, I-1];
%     end
%     dlmwrite(strcat('hmm_eight/eight.hmm.test',file.name), states, ' ');
% end
% system('cat hmm_eight/*.txt > ../../EIGHT.TEST.HMM.SEQ');
% cd ../..

