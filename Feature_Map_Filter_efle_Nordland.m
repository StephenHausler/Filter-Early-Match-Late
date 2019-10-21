function Feature_Map_Filter_efle_Nordland(actLayer,filtActLayer,...
    expNum,HPC,Dataset,saveFolder,settings)

plot_progress = 1;
    
if HPC == 1
    gpuInfo = gpuDevice();  %used to flush the HPC GPU of any prior job data
end

if actLayer >= 19  %specifically for HybridNet
    optFC = 1;
else
    optFC = 0;
end

%batch settings speeds up the filtering at the cost of a slight drop in accuracy
batch_size = settings.batch_size;

%batch option:
batch_option = settings.batch_option; %0=no batch, 1=batch.

%end_index = 1000;

[Qfol,Rfol,GT_file] = Load_Paths(Dataset,HPC);
[filesD,filesQ,dSize,qSize] = Load_Images(Qfol,Rfol,1,1,1); %optional end_index

if HPC == 1
    load('/home/n7542704/MATLAB_2019_Working/Neural_Networks/HybridNet/HybridNet.mat','net');
else
    load('D:\Windows\MATLAB\HybridNet\HybridNet.mat','net');
end

imageCounter = 0; 

%Image processing method adjustable settings:
Initial_crop = settings.initial_crop;  %crop amount in pixels top, bottom, left, right
Initial_crop = Initial_crop + 1;    %only needed because MATLAB starts arrays from index 1.
Im_Resize = net.Layers(1).InputSize;
Im_Resize = Im_Resize(1:2);

for i = settings.calibStartIm:settings.calibSpacing:settings.calibEndIm
%%    
    imageCounter = imageCounter + 1;
    fmap_scoring = zeros(400,400);
    
    t1 = tic; %time to process one calibration image
    time1(imageCounter) = 0;  
    
    Im_Q = imread(char(fullfile(filesQ.path,filesQ.fQ{i})));
    sz = size(Im_Q);
    Im_Q = Im_Q((Initial_crop(1):(sz(1)-Initial_crop(2))),Initial_crop(3):(sz(2)-Initial_crop(4)),:);
    Im_Q = imresize(Im_Q,Im_Resize,'lanczos3');  
    
    d_id = i; %The traverses are aligned on Nordland

    Im_D = imread(char(fullfile(filesD.path,filesD.fD{i}))); 
    
    sz = size(Im_D);
    Im_D = Im_D(Initial_crop(1):(sz(1)-Initial_crop(2)),Initial_crop(3):(sz(2)-Initial_crop(4)),:);
    Im_D = imresize(Im_D,Im_Resize,'lanczos3');  
    
    r = d_id + 50;  %one of the negative images is always from the
    %calibration region (i.e. a hard negative).
    
    Im_D2_1 = imread(char(fullfile(filesD.path,filesD.fD{r}))); %a random image somewhere else in the reference traverse
    
    sz = size(Im_D2_1);
    Im_D2_1 = Im_D2_1(Initial_crop(1):(sz(1)-Initial_crop(2)),Initial_crop(3):(sz(2)-Initial_crop(4)),:);  
    Im_D2_1 = imresize(Im_D2_1,Im_Resize,'lanczos3');  
    
    clear r
    r = randi([2200 3500],4,1);  %four more, random, negative images.
    %need to adjust these for the dataset, these are calibrated for
    %Nordland
    
    Im_D2_2 = imread(char(fullfile(filesD.path,filesD.fD{r(1)})));
    Im_D2_2 = Im_D2_2(Initial_crop(1):(sz(1)-Initial_crop(2)),Initial_crop(3):(sz(2)-Initial_crop(4)),:);
    Im_D2_2 = imresize(Im_D2_2,Im_Resize,'lanczos3');
    Im_D2_3 = imread(char(fullfile(filesD.path,filesD.fD{r(2)})));
    Im_D2_3 = Im_D2_3(Initial_crop(1):(sz(1)-Initial_crop(2)),Initial_crop(3):(sz(2)-Initial_crop(4)),:);
    Im_D2_3 = imresize(Im_D2_3,Im_Resize,'lanczos3');
    Im_D2_4 = imread(char(fullfile(filesD.path,filesD.fD{r(3)})));
    Im_D2_4 = Im_D2_4(Initial_crop(1):(sz(1)-Initial_crop(2)),Initial_crop(3):(sz(2)-Initial_crop(4)),:);
    Im_D2_4 = imresize(Im_D2_4,Im_Resize,'lanczos3');
    Im_D2_5 = imread(char(fullfile(filesD.path,filesD.fD{r(4)})));
    Im_D2_5 = Im_D2_5(Initial_crop(1):(sz(1)-Initial_crop(2)),Initial_crop(3):(sz(2)-Initial_crop(4)),:);
    Im_D2_5 = imresize(Im_D2_5,Im_Resize,'lanczos3');  
    
    [aRef_Q] = CNN_Create_Template(net,Im_Q,actLayer,optFC); 
    [aRef_D] = CNN_Create_Template(net,Im_D,actLayer,optFC);
    [aRef_D2_1] = CNN_Create_Template(net,Im_D2_1,actLayer,optFC);
    [aRef_D2_2] = CNN_Create_Template(net,Im_D2_2,actLayer,optFC);
    [aRef_D2_3] = CNN_Create_Template(net,Im_D2_3,actLayer,optFC);
    [aRef_D2_4] = CNN_Create_Template(net,Im_D2_4,actLayer,optFC);
    [aRef_D2_5] = CNN_Create_Template(net,Im_D2_5,actLayer,optFC);
    
    refDistSame = pdist2(aRef_Q,aRef_D,'euclidean');       %same location, different time of day
    refDistDiff_1 = pdist2(aRef_D,aRef_D2_1,'euclidean');  %different location, same time of day
    refDistDiff_2 = pdist2(aRef_D,aRef_D2_2,'euclidean');  %different location, same time of day
    refDistDiff_3 = pdist2(aRef_D,aRef_D2_3,'euclidean');  %different location, same time of day
    refDistDiff_4 = pdist2(aRef_D,aRef_D2_4,'euclidean');  %different location, same time of day
    refDistDiff_5 = pdist2(aRef_D,aRef_D2_5,'euclidean');  %different location, same time of day

    refDistDiff = (refDistDiff_1 + refDistDiff_2 + refDistDiff_3 + ...
        refDistDiff_4 + refDistDiff_5)/5;
    
    diffRef(imageCounter) = refDistDiff - refDistSame;

    not_Min = 0;
    fCounter = 1;

    clear fmap_track
    clear distSameStore
    clear distDiffStore

    oldlayers = net.Layers;
    totalFeats = sum(oldlayers((filtActLayer-1),1).NumFilters); %previous conv layer on Hnet/Anet
    numFeats = totalFeats;
    trueFmapPos = 1:totalFeats;
    
    act_Q = activations(net, Im_Q, filtActLayer,'OutputAs','channels','ExecutionEnvironment','gpu'); 
    act_D = activations(net, Im_D, filtActLayer,'OutputAs','channels','ExecutionEnvironment','gpu'); 
    act_D2_1 = activations(net, Im_D2_1, filtActLayer,'OutputAs','channels','ExecutionEnvironment','gpu'); 
    act_D2_2 = activations(net, Im_D2_2, filtActLayer,'OutputAs','channels','ExecutionEnvironment','gpu'); 
    act_D2_3 = activations(net, Im_D2_3, filtActLayer,'OutputAs','channels','ExecutionEnvironment','gpu'); 
    act_D2_4 = activations(net, Im_D2_4, filtActLayer,'OutputAs','channels','ExecutionEnvironment','gpu'); 
    act_D2_5 = activations(net, Im_D2_5, filtActLayer,'OutputAs','channels','ExecutionEnvironment','gpu'); 
    sz = size(act_Q);

    %split neural network into half
    halflayer2(1,1) = imageInputLayer(sz,'Name','input','Normalization','none');
    
    lcount = 2;
    for k = (filtActLayer+1):length(oldlayers)
        halflayer2(lcount,1) = oldlayers(k);
        lcount = lcount + 1;
    end
    
    newnet = assembleNetwork(halflayer2);

    %prepare copy acts for zeroing per iteration
    newact_Q = act_Q;
    newact_D = act_D;
    newact_D2_1 = act_D2_1;
    newact_D2_2 = act_D2_2;
    newact_D2_3 = act_D2_3;
    newact_D2_4 = act_D2_4;
    newact_D2_5 = act_D2_5;
%%
    if plot_progress == 1
        figure
        plot_iter = 0;
    end
    
    timeCounter = 0;

    while not_Min == 0      %Greedy search algorithm
        if plot_progress == 1
            plot_iter = plot_iter + 1;
            
            ImArray = cat(4,newact_Q,newact_D,newact_D2_1,newact_D2_2,newact_D2_3,newact_D2_4,newact_D2_5);
            
            if optFC == 1
                [Vectors] = CNN_Create_Template_Array_FC(newnet,ImArray,(actLayer-filtActLayer+1));
            else
                [Vectors] = CNN_Create_Template_Array(newnet,ImArray,(actLayer-filtActLayer+1));
            end
                
            vector_Q = Vectors(1,:);vector_D = Vectors(2,:);
            vector_D2_1 = Vectors(3,:);vector_D2_2 = Vectors(4,:);
            vector_D2_3 = Vectors(5,:);vector_D2_4 = Vectors(6,:);
            vector_D2_5 = Vectors(7,:);

            distSame_p(plot_iter) = pdist2(vector_Q,vector_D,'euclidean'); %same location, different time of day
            distDiff_1_p = pdist2(vector_D,vector_D2_1,'euclidean');  %different location, same time of day
            distDiff_2_p = pdist2(vector_D,vector_D2_2,'euclidean');  %different location, same time of day
            distDiff_3_p = pdist2(vector_D,vector_D2_3,'euclidean');  %different location, same time of day
            distDiff_4_p = pdist2(vector_D,vector_D2_4,'euclidean');  %different location, same time of day
            distDiff_5_p = pdist2(vector_D,vector_D2_5,'euclidean');  %different location, same time of day
            distDiff_p(plot_iter) = (distDiff_1_p + distDiff_2_p + distDiff_3_p + distDiff_4_p + distDiff_5_p)/5;
            
            xplot = 0:4:((plot_iter-1)*4);
            
            plot(xplot,distSame_p);
            hold on
            plot(xplot,distDiff_p);
            ylabel('L2 Distance');
            xlabel('Filter Iteration Count (Number of Maps Removed)');
            title(sprintf('L2 Distance over Amount of Filtering with Filter Layer %d and Extract Layer %d'...
                ,filtActLayer,actLayer));
            legend('Same Location, Different Time of Day','Different Locations, Same Time of Day');
            set(gca,'FontSize',18);
            hold off
            drawnow; 
        end
        
        t2 = tic; %time to loop one iteration of Greedy filter algorithm
        timeCounter = timeCounter + 1;
        time2(timeCounter) = 0;  
        
        summed_scoring = sum(fmap_scoring);
        for j = 1:numFeats            
            j2 = Remember(j,summed_scoring,totalFeats);
            
            act_Q_filt = newact_Q;
            act_Q_filt(:,:,j2) = 0;
            
            act_D_filt = newact_D;
            act_D_filt(:,:,j2) = 0;
            
            act_D2_filt_1 = newact_D2_1;
            act_D2_filt_1(:,:,j2) = 0;
            
            act_D2_filt_2 = newact_D2_2;
            act_D2_filt_2(:,:,j2) = 0;
            
            act_D2_filt_3 = newact_D2_3;
            act_D2_filt_3(:,:,j2) = 0;
            
            act_D2_filt_4 = newact_D2_4;
            act_D2_filt_4(:,:,j2) = 0;
            
            act_D2_filt_5 = newact_D2_5;
            act_D2_filt_5(:,:,j2) = 0;  
            
            ImArray = cat(4,act_Q_filt,act_D_filt,act_D2_filt_1,act_D2_filt_2,act_D2_filt_3,act_D2_filt_4,act_D2_filt_5);

            %add if statement to check if fully connected layer...
            if optFC == 1
                [Vectors] = CNN_Create_Template_Array_FC(newnet,ImArray,(actLayer-filtActLayer+1));
            else
                [Vectors] = CNN_Create_Template_Array(newnet,ImArray,(actLayer-filtActLayer+1));
            end
                
            vector_Q = Vectors(1,:);vector_D = Vectors(2,:);
            vector_D2_1 = Vectors(3,:);vector_D2_2 = Vectors(4,:);
            vector_D2_3 = Vectors(5,:);vector_D2_4 = Vectors(6,:);
            vector_D2_5 = Vectors(7,:);

            distSame(j) = pdist2(vector_Q,vector_D,'euclidean'); %same location, different time of day
            distDiff_1 = pdist2(vector_D,vector_D2_1,'euclidean');  %different location, same time of day
            distDiff_2 = pdist2(vector_D,vector_D2_2,'euclidean');  %different location, same time of day
            distDiff_3 = pdist2(vector_D,vector_D2_3,'euclidean');  %different location, same time of day
            distDiff_4 = pdist2(vector_D,vector_D2_4,'euclidean');  %different location, same time of day
            distDiff_5 = pdist2(vector_D,vector_D2_5,'euclidean');  %different location, same time of day
            distDiff(j) = (distDiff_1 + distDiff_2 + distDiff_3 + distDiff_4 + distDiff_5)/5;
        end
        diff = distDiff - distSame;
        %shouldnt use abs here because want distDiff to be greater than distSame!

        [maxval,worst_Fmap] = max(diff); %which worst feature map to remove results in
        %the greatest distance between the same location and a different
        %location while minimising the distance between the same location at 
        %different times of day.
        
        %to speed things up, remove batch_size worst maps at each iteration...
        if batch_option == 1
            for bb = 2:batch_size
                diff(worst_Fmap(bb-1)) = -1e3;
                [~,worst_Fmap(bb)] = max(diff);
            end
        end

        clear distSame
        clear distDiff
        clear diff
        
        if fCounter > (totalFeats/2)
            not_Min = 1;
        end 
        
        if not_Min == 0
            if batch_option == 1
                for k = 1:batch_size
                    fmap_track(fCounter) = worst_Fmap(k);
                    fCounter = fCounter + 1;
                end
                for k = (fCounter-batch_size):(fCounter-1) %1 to 4 wrt current fCounter
                    try
                        fmap_scoring(k,trueFmapPos(fmap_track(k))) = 1;
                    catch
                        warning('Out of bounds!');
                        break
                    end
                    trueFmapPos(fmap_track(k)) = 0;
                end
                [~,~,trueFmapPos] = find(trueFmapPos);
            else
                fmap_track(fCounter) = worst_Fmap;
                try
                    fmap_scoring(k,trueFmapPos(fmap_track(fCounter))) = 1;
                catch
                    warning('Out of bounds!');
                    break
                end
                trueFmapPos(fmap_track(fCounter)) = 0;
                
                [~,~,trueFmapPos] = find(trueFmapPos);
                fCounter = fCounter + 1;
            end
            
            maxvalstore(imageCounter,fCounter-1) = maxval;
            
            newact_Q = act_Q;
            newact_D = act_D;
            newact_D2_1 = act_D2_1;
            newact_D2_2 = act_D2_2;
            newact_D2_3 = act_D2_3;
            newact_D2_4 = act_D2_4;
            newact_D2_5 = act_D2_5;
            for j = 1:length(fmap_scoring)
                for k = 1:(fCounter-1)
                    if fmap_scoring(k,j) == 1
                        newact_Q(:,:,j) = 0;
                        newact_D(:,:,j) = 0;
                        newact_D2_1(:,:,j) = 0;
                        newact_D2_2(:,:,j) = 0;
                        newact_D2_3(:,:,j) = 0;
                        newact_D2_4(:,:,j) = 0;
                        newact_D2_5(:,:,j) = 0;
                    end
                end   
            end
            if batch_option == 1
                numFeats = numFeats - batch_size;
            else    
                numFeats = numFeats - 1;
            end    
        end
        time2(timeCounter) = time2(timeCounter) + toc(t2);
    end
    %remove the 0s from maxvalstore... (only occurs if using batches)
    clear maxvalstore2
    try
        maxvalstore2 = maxvalstore(imageCounter,find(maxvalstore(imageCounter,:)));
    catch
        warning('Error');
    end    
    [~,globalmax] = max(maxvalstore2);
    globalmax = globalmax*batch_size;
    final_scores(imageCounter,:) = sum(fmap_scoring(1:globalmax,:));
    track_final_map_counts(imageCounter) = globalmax;
    
    avTime2 = mean(time2);

    save(sprintf('RunStatus_%d.mat',expNum),'imageCounter','avTime2');
    
    time1(imageCounter) = time1(imageCounter) + toc(t1);
end
final_scores = final_scores(:,1:totalFeats);
finalScores(1,:) = 1:totalFeats;
finalScores(2,:) = sum(final_scores,1);

mean_map_count = mean(track_final_map_counts);
mean_map_count = floor(mean_map_count);

saveFinalScores = finalScores;

for k = 1:mean_map_count
    [~,i] = max(finalScores(2,:));
    finalMaps(1,k) = i;
    finalScores(2,i) = -10000;
end

finalScores = saveFinalScores;
totalNumCalibImages = imageCounter;

avTime1 = mean(time1);

save([saveFolder Dataset '_' sprintf('Exp%d',expNum) '_' sprintf('finalMaps_filtLayer_%d_extractLayer_%d.mat',...
    filtActLayer,actLayer)],'finalMaps','mean_map_count','finalScores',...
    'maxvalstore','diffRef','totalNumCalibImages','avTime1','avTime2');
end

function jj = Remember(mapID,fmapScoring,totalFeats)
    for i = 1:totalFeats
        if ((fmapScoring(i) == 1) && (i <= mapID))
            mapID = mapID + 1;
        end
    end    
    jj = mapID;
end






