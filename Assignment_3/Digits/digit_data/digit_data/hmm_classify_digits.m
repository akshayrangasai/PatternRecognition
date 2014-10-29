load two.txt
load three.txt
load seven.txt
load six.txt
load eight.txt

[~, C] = kmeans([two; three; seven; six; eight], 20);

files1 = dir('two/train/*.txt');
files3 = dir('three/train/*.txt');
files5 = dir('seven/train/*.txt');
files6 = dir('six/train/*.txt');
files8 = dir('eight/train/*.txt');

cd two/train/
mkdir hmm_two
for file = files1'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_two/two.hmm.train',file.name), states, ' ');
end
system('cat hmm_two/*.txt > ../../two.TRAIN.HMM.SEQ');
cd ../..

cd three/train/
mkdir hmm_three
for file = files3'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_three/three.hmm.train',file.name), states, ' ');
end
system('cat hmm_three/*.txt > ../../THREE.TRAIN.HMM.SEQ');
cd ../..

cd seven/train/
mkdir hmm_seven
for file = files5'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_seven/seven.hmm.train',file.name), states, ' ');
end
system('cat hmm_seven/*.txt > ../../seven.TRAIN.HMM.SEQ');
cd ../..

cd six/train/
mkdir hmm_six
for file = files6'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_six/six.hmm.train',file.name), states, ' ');
end
system('cat hmm_six/*.txt > ../../SIX.TRAIN.HMM.SEQ');
cd ../..

cd eight/train/
mkdir hmm_eight
for file = files8'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_eight/eight.hmm.train',file.name), states , ' ');
end
system('cat hmm_eight/*.txt > ../../EIGHT.TRAIN.HMM.SEQ');
cd ../..

%-----------_TEST
files1 = dir('two/test/*.txt');
files3 = dir('three/test/*.txt');
files5 = dir('seven/test/*.txt');
files6 = dir('six/test/*.txt');
files8 = dir('eight/test/*.txt');

cd two/test/
mkdir hmm_two
for file = files1'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_two/two.hmm.test',file.name), states, ' ');
end
system('cat hmm_two/*.txt > ../../two.TEST.HMM.SEQ');
cd ../..

cd three/test/
mkdir hmm_three
for file = files3'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_three/three.hmm.test',file.name), states, ' ');
end
system('cat hmm_three/*.txt > ../../THREE.TEST.HMM.SEQ');
cd ../..

cd seven/test/
mkdir hmm_seven
for file = files5'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_seven/seven.hmm.test',file.name), states, ' ');
end
system('cat hmm_seven/*.txt > ../../seven.TEST.HMM.SEQ');
cd ../..

cd six/test/
mkdir hmm_six
for file = files6'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_six/six.hmm.test',file.name), states, ' ');
end
system('cat hmm_six/*.txt > ../../SIX.TEST.HMM.SEQ');
cd ../..

cd eight/test/
mkdir hmm_eight
for file = files8'
    tmp = load(file.name);
    states = [];
    for i = 1:length(tmp)
        ss = (C - repmat(tmp(i,:), 20, 1)).^2;
        d = sum(ss,2);
        [~,I] = min(d);
        states = [states, I-1];
    end
    dlmwrite(strcat('hmm_eight/eight.hmm.test',file.name), states, ' ');
end
system('cat hmm_eight/*.txt > ../../EIGHT.TEST.HMM.SEQ');
cd ../..