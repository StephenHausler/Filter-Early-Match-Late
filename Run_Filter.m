clc
clear
close all

actArray = [7 11 13 15 17 20];  %for Hnet (conv 2,3,4,5,6 ReLu layers and FC 1 ReLu layer)
filtActArray = [7 11 13 15 17]; %(conv 2,3,4,5,6 ReLu layers)

featExtractLayer = actArray(4);
featFilterLayer = filtActArray(1);

saveFolder = 'HybridNet_Filters_Nordland/';

Dataset = 'Nordland';

expNum = 1;

saveName = [saveFolder Dataset '_' sprintf('Exp%d',expNum) '_' sprintf(...
    'finalMaps_filtLayer_%d_extractLayer_%d.mat',featFilterLayer,featExtractLayer)];

HPC = 0;

%for Nordland:
settings.initial_crop = [0 0 0 0];
settings.batch_option = 1;
settings.batch_size = 4;
settings.calibStartIm = 1470;
settings.calibEndIm = 1715;
settings.calibSpacing = 5;
settings.runStartIm = 1769;
settings.runEndIm = 3769;
settings.runSpacing = 1;

[recall,precision] = Filter_Place_Recognition...
    (saveName,featExtractLayer,featFilterLayer,Dataset,HPC,settings,expNum);




