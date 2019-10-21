function [template,heatmap] = CNN_Create_Template_Extended(net,Im,actLayer1,optFC)
%Need to have net pre-defined as a global variable

if optFC == 1
    act = activations(net, Im, actLayer1,'OutputAs','channels','ExecutionEnvironment','gpu');
    sz1 = size(act);
    template = reshape(act,[1 sz1(3)]);
else
    act = activations(net, Im, actLayer1,'OutputAs','channels','ExecutionEnvironment','gpu');
    
    %try something very different:
%     for k = 1:13
%         act(1,k,:)=0;
%         act(k,1,:)=0;
%         act(13,k,:)=0;
%         act(k,13,:)=0;
%     end

%    sumAct = sum(act,3);
%    act = act./sumAct;

    sz1 = size(act); 

    act1 = reshape(act,[sz1(1) sz1(2) 1 sz1(3)]);

    sh11 = ceil(sz1(1)/2); sh12 = ceil(sz1(2)/2);

    sum_array=0;

    for j = 1:sz1(3)
        sum_array(1,j) = max(max(act1(:,:,1,j)));
        sum_array(2,j) = max(max(act1(1:sh11,1:sh12,1,j)));
        sum_array(3,j) = max(max(act1(1:sh11,sh12:sz1(2),1,j)));        
        sum_array(4,j) = max(max(act1(sh11:sz1(1),1:sh12,1,j)));
        sum_array(5,j) = max(max(act1(sh11:sz1(1),sh12:sz1(2),1,j)));
    end

    sz1 = size(sum_array);

    template = reshape(sum_array,[1 sz1(1)*sz1(2)]);
    
    hact1 = act1;
    sz1 = size(hact1);
    
    %add total energy based heatmap selection instead:
    
    %original:
    
    mean_act1 = mean(max(max(hact1(:,:,1,:))));
    t=1;
    %rtmp = max in a row.
    %ctmp = max in a column.
    for j = 1:sz1(4)
        [mctmp,rtmp] = max(hact1(:,:,1,j));
        [mrtmp,ctmp] = max(mctmp);
        
        %if myhact == 0
        if mrtmp < mean_act1
            
        else
            c(t) = ctmp;
            r(t) = 14 - rtmp(ctmp);
            t=t+1;
        end
    end
    heatmap = zeros(13);
    
    for t = 1:length(r)
        heatmap(r(t),c(t)) = heatmap(r(t),c(t))+1;
    end
end

end




