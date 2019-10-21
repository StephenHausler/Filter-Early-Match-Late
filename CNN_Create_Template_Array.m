function [template] = CNN_Create_Template_Array(net,ImArray,actLayer1)
%Need to have net pre-defined as a global variable

act = activations(net, ImArray, actLayer1,'OutputAs','channels','ExecutionEnvironment','gpu');

act1 = act(:,:,:,1); act2 = act(:,:,:,2); act3 = act(:,:,:,3);
act4 = act(:,:,:,4); act5 = act(:,:,:,5); act6 = act(:,:,:,6); act7 = act(:,:,:,7);

sz1 = size(act1);  %all same size

act1 = reshape(act1,[sz1(1) sz1(2) 1 sz1(3)]);
act2 = reshape(act2,[sz1(1) sz1(2) 1 sz1(3)]);
act3 = reshape(act3,[sz1(1) sz1(2) 1 sz1(3)]);
act4 = reshape(act4,[sz1(1) sz1(2) 1 sz1(3)]);
act5 = reshape(act5,[sz1(1) sz1(2) 1 sz1(3)]);
act6 = reshape(act6,[sz1(1) sz1(2) 1 sz1(3)]);
act7 = reshape(act7,[sz1(1) sz1(2) 1 sz1(3)]);

sh11 = ceil(sz1(1)/2); sh12 = ceil(sz1(2)/2);

sum_array1 = zeros(1,sz1(3));
sum_array2 = zeros(1,sz1(3));
sum_array3 = zeros(1,sz1(3));
sum_array4 = zeros(1,sz1(3));
sum_array5 = zeros(1,sz1(3));
sum_array6 = zeros(1,sz1(3));
sum_array7 = zeros(1,sz1(3));

for j = 1:sz1(3)
    sum_array1(1,j) = max(max(act1(:,:,1,j)));
    sum_array1(2,j) = max(max(act1(1:sh11,1:sh12,1,j)));
    sum_array1(3,j) = max(max(act1(1:sh11,sh12:sz1(2),1,j)));        
    sum_array1(4,j) = max(max(act1(sh11:sz1(1),1:sh12,1,j)));
    sum_array1(5,j) = max(max(act1(sh11:sz1(1),sh12:sz1(2),1,j)));
    
    sum_array2(1,j) = max(max(act2(:,:,1,j)));
    sum_array2(2,j) = max(max(act2(1:sh11,1:sh12,1,j)));
    sum_array2(3,j) = max(max(act2(1:sh11,sh12:sz1(2),1,j)));        
    sum_array2(4,j) = max(max(act2(sh11:sz1(1),1:sh12,1,j)));
    sum_array2(5,j) = max(max(act2(sh11:sz1(1),sh12:sz1(2),1,j)));
    
    sum_array3(1,j) = max(max(act3(:,:,1,j)));
    sum_array3(2,j) = max(max(act3(1:sh11,1:sh12,1,j)));
    sum_array3(3,j) = max(max(act3(1:sh11,sh12:sz1(2),1,j)));
    sum_array3(4,j) = max(max(act3(sh11:sz1(1),1:sh12,1,j)));
    sum_array3(5,j) = max(max(act3(sh11:sz1(1),sh12:sz1(2),1,j)));
    
    sum_array4(1,j) = max(max(act4(:,:,1,j)));
    sum_array4(2,j) = max(max(act4(1:sh11,1:sh12,1,j)));
    sum_array4(3,j) = max(max(act4(1:sh11,sh12:sz1(2),1,j)));
    sum_array4(4,j) = max(max(act4(sh11:sz1(1),1:sh12,1,j)));
    sum_array4(5,j) = max(max(act4(sh11:sz1(1),sh12:sz1(2),1,j)));
    
    sum_array5(1,j) = max(max(act5(:,:,1,j)));
    sum_array5(2,j) = max(max(act5(1:sh11,1:sh12,1,j)));
    sum_array5(3,j) = max(max(act5(1:sh11,sh12:sz1(2),1,j)));
    sum_array5(4,j) = max(max(act5(sh11:sz1(1),1:sh12,1,j)));
    sum_array5(5,j) = max(max(act5(sh11:sz1(1),sh12:sz1(2),1,j)));
    
    sum_array6(1,j) = max(max(act6(:,:,1,j)));
    sum_array6(2,j) = max(max(act6(1:sh11,1:sh12,1,j)));
    sum_array6(3,j) = max(max(act6(1:sh11,sh12:sz1(2),1,j)));
    sum_array6(4,j) = max(max(act6(sh11:sz1(1),1:sh12,1,j)));
    sum_array6(5,j) = max(max(act6(sh11:sz1(1),sh12:sz1(2),1,j)));
    
    sum_array7(1,j) = max(max(act7(:,:,1,j)));
    sum_array7(2,j) = max(max(act7(1:sh11,1:sh12,1,j)));
    sum_array7(3,j) = max(max(act7(1:sh11,sh12:sz1(2),1,j)));
    sum_array7(4,j) = max(max(act7(sh11:sz1(1),1:sh12,1,j)));
    sum_array7(5,j) = max(max(act7(sh11:sz1(1),sh12:sz1(2),1,j)));
end

sz1 = size(sum_array1);
    
template1 = reshape(sum_array1,[1 sz1(1)*sz1(2)]);
template2 = reshape(sum_array2,[1 sz1(1)*sz1(2)]);
template3 = reshape(sum_array3,[1 sz1(1)*sz1(2)]);
template4 = reshape(sum_array4,[1 sz1(1)*sz1(2)]);
template5 = reshape(sum_array5,[1 sz1(1)*sz1(2)]);
template6 = reshape(sum_array6,[1 sz1(1)*sz1(2)]);
template7 = reshape(sum_array7,[1 sz1(1)*sz1(2)]);

template = cat(1,template1,template2,template3,template4,template5,...
    template6,template7);

end




