clc
clear
close all

actArray = [7 11 13 15 17 20];  %for Hnet (conv 2,3,4,5,6 ReLu layers and FC 1 ReLu layer)
filtActArray = [3 7 11 13 15 17]; %(conv 1,2,3,4,5,6 ReLu layers)

featExtractLayer = actArray(5);
featFilterLayer = filtActArray(2);

%Dataset = 'Oxford';
Dataset = 'Nordland';

%saveFolder = 'HybridNet_Filters_Oxford/';
saveFolder = 'HybridNet_Filters_Nordland/';
mkdir(saveFolder);

expNum = 1;

HPC = 0;

%for Nordland:
settings.initial_crop = [0 0 0 0];
settings.batch_option = 1;
settings.batch_size = 4;
settings.calibStartIm = 1470;
settings.calibEndIm = 1715;
%settings.calibEndIm = 1475;
settings.calibSpacing = 5;

%for Oxford Robotcar:
% settings.initial_crop = [20 140 0 0];
% settings.batch_option = 1;
% settings.batch_size = 4;
% settings.calibStartIm = 92;
% settings.calibEndIm = 1072;
% settings.calibSpacing = 20;

Feature_Map_Filter_efle_Nordland(featExtractLayer,featFilterLayer,...
    expNum,HPC,Dataset,saveFolder,settings);




