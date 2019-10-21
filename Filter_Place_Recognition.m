function [recall,precision] = Filter_Place_Recognition...
    (saveName,actLayer,filtActLayer,Dataset,HPC,settings,expNum)

plot_progress = 1;
load_precomputed = 1;

if HPC == 1
    gpuInfo = gpuDevice();
    load('/home/n7542704/MATLAB_2019_Working/Neural_Networks/HybridNet/HybridNet.mat','net');
    Red_Box = imread("/home/n7542704/MATLAB_2019_Working/Continuous_Feature_Map_Filtering/Code_and_PBS_Scripts/Red_Box.jpg");
else
    load('D:\Windows\MATLAB\HybridNet\HybridNet.mat','net');
    Red_Box = imread("F:\MATLAB_2019_Working\Continuous_Feature_Map_Filtering\Code_and_PBS_Scripts\Red_Box.jpg");
end

if actLayer >= 19  %specifically for HybridNet
    optFC = 1;
else
    optFC = 0;
end

removeMapCount = 0;

Frame_skip = settings.runSpacing;
imStartR = settings.runStartIm;
imStartQ = settings.runStartIm;
imEndR = settings.runEndIm;
imEndQ = settings.runEndIm; 

[Qfol,Rfol,GT_file] = Load_Paths(Dataset,HPC);
[filesD,filesQ,dSize,qSize] = Load_Images(Qfol,Rfol,imStartR,imStartQ...
    ,Frame_skip,imEndR,imEndQ); %optional end_index

%Image processing method adjustable settings:
Initial_crop = settings.initial_crop;  %crop amount in pixels top, bottom, left, right
Initial_crop = Initial_crop + 1;    %only needed because MATLAB starts arrays from index 1.
Im_Resize = net.Layers(1).InputSize;
Im_Resize = Im_Resize(1:2);

%load the filter variables
load(saveName,'finalMaps','mean_map_count','finalScores','maxvalstore','diffRef','totalNumCalibImages'); 

filterThresh = 0.66*totalNumCalibImages;
for k = 1:length(finalScores)
    if finalScores(2,k) > filterThresh
        removeMapCount = removeMapCount + 1;
    end
end

thresh = [0.0001 0.0005 0.001 0.0015 0.002 0.0025 0.003 0.0035 0.004 0.0045...
    0.005 0.0055 0.0065 0.007 0.0075 0.008 0.0085 0.009  0.0095 0.01 0.011...
    0.012 0.013 0.014 0.015 0.0175 0.02 0.025 0.03 0.035 0.04 0.045 0.05 ...
    0.06 0.07 0.08 0.09 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3...
    0.32 0.34 0.36 0.38 0.4 0.42 0.44 0.46 0.48 0.5 0.6 0.7 0.8 0.9 1.0]; 

obsThresh = 0.1;
epsilon = 0.001;
Rwindow = 10;           %Change this to reflect the approximate distance 
%between frames and the localisation accuracy required.

%Zeroing Variables
recall_count = zeros(1,length(thresh));
error_count = zeros(1,length(thresh));
false_negative_count = zeros(1,length(thresh));

%Reference Dataset:
%--------------------------------------------------------------------------
%start setting up new layers post filtering:
newlayers = net.Layers;
%totalFeats = sum(newlayers((filtActLayer-1),1).NumFilters); 

for k = 1:removeMapCount
    [~,i] = max(finalScores(2,:));
    newFinalMaps(1,k) = i;
    finalScores(2,i) = -10000;
end

removedMaps = newFinalMaps;

for i = 1:length(removedMaps)    
    newlayers((filtActLayer-1),1).Weights(:,:,:,removedMaps(i)) = 0;  
    newlayers((filtActLayer-1),1).Bias(:,:,removedMaps(i)) = 0;
end    
newnet = assembleNetwork(newlayers);

if load_precomputed == 1
    load(['Ref_Nord_Dbase_Feats_actLayer_' num2str(actLayer) '_' '11-Oct-2019'],...
        'Database_Feats','Database_Heatmapsave');    
    load(['Query_Nord_Dbase_Feats_actLayer_' num2str(actLayer) '_' '11-Oct-2019'],...
        'Query_Feats','Query_Heatmapsave');
else
for i = 1:dSize
    Im = imread(char(fullfile(filesD.path,filesD.fD{i})));
    sz = size(Im);
    Im = Im((Initial_crop(1):(sz(1)-Initial_crop(2))),Initial_crop(3):(sz(2)-Initial_crop(4)),:);
    Im1 = imresize(Im,Im_Resize,'lanczos3');  
    
    if optFC == 1
        [template] = CNN_Create_Template_Extended(newnet,Im1,actLayer,optFC);
    else   
        [template,heatmap] = CNN_Create_Template_Extended(newnet,Im1,actLayer,optFC);
        Database_Heatmapsave(:,:,i) = heatmap;
    end
    
    Database_Feats(i,:) = template;
end

save(['Ref_Nord_Dbase_Feats_actLayer_' num2str(actLayer) '_' date()],...
    'Database_Feats','Database_Heatmapsave');

for i = 1:qSize
    Im = imread(char(fullfile(filesQ.path,filesQ.fQ{i})));
    sz = size(Im);
    Im = Im((Initial_crop(1):(sz(1)-Initial_crop(2))),Initial_crop(3):(sz(2)-Initial_crop(4)),:);
    Im1 = imresize(Im,Im_Resize,'lanczos3');  
    
    if optFC == 1
        [template] = CNN_Create_Template_Extended(newnet,Im1,actLayer,optFC);
    else   
        [template,heatmap] = CNN_Create_Template_Extended(newnet,Im1,actLayer,optFC);
        Query_Heatmapsave(:,:,i) = heatmap;
    end
    
    Query_Feats(i,:) = template;
end

save(['Query_Nord_Dbase_Feats_actLayer_' num2str(actLayer) '_' date()],...
    'Query_Feats','Query_Heatmapsave');

end

O = zeros(dSize,1);

%now to recognition process
%start with place matching loop, 1:qSize
%then run a second plotting loop.

%which is better: L2 or cosine?
for i = 1:qSize
    D = pdist2(Query_Feats(i,:),Database_Feats,'cosine');
    mx1 = max(D);
    df1 = mx1 - min(D);
    %normalise such that 0.999 is best and 0.001 is worst
    for k = 1:dSize
        O_diff = ((mx1 - D(k))/df1)-epsilon;  %normalise to range 0.001 to 0.999    
        if O_diff < obsThresh %only used to ensure min is not -0.001
            O(k) = epsilon; 
        else
            O(k) = O_diff;
        end 
    end
    [minval,id] = max(O);
    window = max(1, id-Rwindow):min(length(O), id+Rwindow);
    not_window = setxor(1:length(O), window);
    min_value_2nd = max(O(not_window));
    quality = log(minval) / log(min_value_2nd);
    
    matched_image(i) = id;
    
    %using id and quality, calculate precision and recall counts
    if (GT_file.GPSMatrix(imStartR + id,imStartQ + i) == 1)
        recall_by_image(i) = 1;
    else
        recall_by_image(i) = 0;
    end
    for thresh_counter = 1:length(thresh)
        if quality > thresh(thresh_counter)
            false_negative_count(thresh_counter) = false_negative_count(thresh_counter) + 1;
        else
            if recall_by_image(i) == 1
                recall_count(thresh_counter) = recall_count(thresh_counter) + 1;
            else
                error_count(thresh_counter) = error_count(thresh_counter) + 1;
            end
        end
    end
end

for thresh_counter = 1:length(thresh)
    %Recall = true positives / (true positives + false negatives)
    recall(thresh_counter) = recall_count(thresh_counter)/(recall_count(thresh_counter) + false_negative_count(thresh_counter));
    %Precision = true positives / (true positives + false positives)
    precision(thresh_counter) = recall_count(thresh_counter)/(recall_count(thresh_counter) + error_count(thresh_counter));
end

recall_rate = sum(recall_by_image)/qSize;

save(['Nord_Results_actLayer_' num2str(actLayer) '_' date()],...
    'recall_by_image','recall_rate','recall','precision');

OrigDBHMS = Database_Heatmapsave;
clear Database_Heatmapsave;
load('Ref_Nord_Dbase_Feats_actLayer_15_10-Oct-2019_orig_nofilter.mat','Database_Heatmapsave');
Nofilt_DBHMS = Database_Heatmapsave;
Database_Heatmapsave = OrigDBHMS;

if plot_progress == 1
    for i = 1:qSize
        ImQ = imread(char(fullfile(filesQ.path,filesQ.fQ{i})));
        ImD = imread(char(fullfile(filesD.path,filesD.fD{matched_image(i)})));
        ImD_gt = imread(char(fullfile(filesD.path,filesD.fD{i})));
        
        ImQ = imresize(ImQ,[221 221],'lanczos3');
        ImD = imresize(ImD,[221 221],'lanczos3');
        ImD_gt = imresize(ImD_gt,[221 221],'lanczos3');
        
        subplot(1,3,1,'replace');
        QFH = Query_Heatmapsave(:,:,i);
        QFHL = imresize(QFH,17);
        QFHL = QFHL - min(min(QFHL));
        trp = 1./QFHL;
        trp = trp.*3;
        h = imagesc(QFHL);
        hold on
        I = imshow(ImQ);
        hold off  
        set(I,'AlphaData',trp);
        title('Current Scene Activation HeatMap','fontsize',20);

        subplot(1,3,2,'replace');
        RFH = Database_Heatmapsave(:,:,matched_image(i));
        RFHL = imresize(RFH,17);
        RFHL = RFHL - min(min(RFHL));
        trp = 1./RFHL;
        trp = trp.*3;
        if recall_by_image(i) == 0
            h = imagesc(RFHL);
            hold on
            I = imshow(ImD);
            sz = size(ImD);
            rectangle('Position',[1,1,sz(2)-1,sz(1)-1],...
                'EdgeColor', 'r',...
                'LineWidth', 5,...
                'LineStyle','-')
            hold off
        else
            h = imagesc(RFHL);
            hold on
            I = imshow(ImD);
            hold off
        end  
        set(I,'AlphaData',trp);
        title('Matched Scene Activation HeatMap','fontsize',20);
        
%need to edit the below, cause currently only works on Nordland
%         subplot(1,3,3,'replace');
%         RFH = Database_Heatmapsave(:,:,i);
%         RFHL = imresize(RFH,17);
%         RFHL = RFHL - min(min(RFHL));
%         trp = 1./RFHL;
%         trp = trp.*3;
%         h = imagesc(RFHL);
%         hold on
%         I = imshow(ImD_gt);
%         hold off  
%         set(I,'AlphaData',trp);
%         title('Ground Truth Match Activation HeatMap','fontsize',20);  

        subplot(1,3,3,'replace');
        RFH = Nofilt_DBHMS(:,:,matched_image(i));
        RFHL = imresize(RFH,17);
        RFHL = RFHL - min(min(RFHL));
        trp = 1./RFHL;
        trp = trp.*3;
        h = imagesc(RFHL);
        hold on
        I = imshow(ImD_gt);
        hold off  
        set(I,'AlphaData',trp);
        title('No Filter Match Activation HeatMap','fontsize',20);  

        drawnow;
        pause(0.2);       
    end
end 
    
end


