   load 'RWD.mat';
   trainData = RWD; 
   m = size(trainData, 1); % number of training examples
   n = size(trainData, 2) - 1; % number of feature dimension
   k = length(unique(trainData(:, end))); % number of classes
   class_matrices = zeros(m,n,k);
   counts = ones(1,k);
  
   for i = 1:m
       class_matrices(counts(1,trainData(i,n+1)),:,trainData(i,n+1)) = trainData(i,1:n);
       counts(1,trainData(i,n+1)) = counts(1,trainData(i,n+1)) + 1;
   end
   
   clf;
   whitebg('white');
   plot(class_matrices(1: counts(1,1)-1,1,1),class_matrices(1:counts(1,1)-1,2,1),'r.')
   hold on
   plot(class_matrices(1:counts(1,2)-1,1,2),class_matrices(1:counts(1,2)-1,2,2),'g.');
   hold on
   plot(class_matrices(1:counts(1,3)-1,1,3),class_matrices(1:counts(1,3)-1,2,3),'b.');
   hold on
   %plot(class_matrices(1:counts(1,4)-1,1,4),class_matrices(1:counts(1,4)-1,2,4),'c.');
   %hold on

   for k=1:3
       [v,d] = eig(class_matrices(1:counts(1,k)-1,:,k)'*class_matrices(1:counts(1,k)-1,:,k));
       mu = mean(class_matrices(1:counts(1,trainData(k,n+1))-1,:,k));
       plot([mu(1) mu(1)+100*v(1,2)],[mu(2) mu(2)+100*v(2,2)],'k-','LineWidth',3);
       hold on
       plot([mu(1) mu(1)+100*v(1,1)],[mu(2) mu(2)+100*v(2,1)],'k-.','LineWidth',3);
       hold on
   end
   