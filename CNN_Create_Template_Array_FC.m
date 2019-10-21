function [template] = CNN_Create_Template_Array_FC(net,ImArray,actLayer1)
%Need to have net pre-defined as a global variable

act = activations(net, ImArray, actLayer1,'ExecutionEnvironment','gpu');

act1 = act(:,:,:,1); act2 = act(:,:,:,2); act3 = act(:,:,:,3);
act4 = act(:,:,:,4); act5 = act(:,:,:,5); act6 = act(:,:,:,6); act7 = act(:,:,:,7);

sz1 = size(act1);  %all same size

act1 = reshape(act1,[1 sz1(3)]);
act2 = reshape(act2,[1 sz1(3)]);
act3 = reshape(act3,[1 sz1(3)]);
act4 = reshape(act4,[1 sz1(3)]);
act5 = reshape(act5,[1 sz1(3)]);
act6 = reshape(act6,[1 sz1(3)]);
act7 = reshape(act7,[1 sz1(3)]);

template = cat(1,act1,act2,act3,act4,act5,...
    act6,act7);

end




