function [Qfol, Rfol, GT_file] = Load_Paths(Dataset, HPC)

if HPC == 1
    if strcmp(Dataset,'Nordland')
        Qfol = '/home/n7542704/D_Drive_Backup/Windows/Nordland/nordland_summer_images';
        Rfol = '/home/n7542704/D_Drive_Backup/Windows/Nordland/nordland_winter_images';
        GT_file = load('/home/n7542704/D_Drive_Backup/Windows/Nordland/Nordland_GPSMatrix.mat');
    elseif strcmp(Dataset,'Berlin')
%         Qfol = '/home/n7542704/Datasets/Berlin_Kudamm/Query_Test';
%         Rfol = '/home/n7542704/Datasets/Berlin_Kudamm/Reference_Test';
%         GT_file = load('/home/n7542704/Datasets/Berlin_Kudamm/Berlin_GPSMatrix_50meters_Test.mat');
%         Qfol = '/home/n7542704/Datasets/Berlin_Kudamm/Query';
%         Rfol = '/home/n7542704/Datasets/Berlin_Kudamm/Reference';
%         GT_file = load('/home/n7542704/Datasets/Berlin_Kudamm/Berlin_GPSMatrix_50meters.mat');
        Qfol = '/home/n7542704/Datasets/Berlin_Kudamm/Query_Test';
        Rfol = '/home/n7542704/Datasets/Berlin_Kudamm/Reference_Test';
        GT_file = load('/home/n7542704/Datasets/Berlin_Kudamm/Berlin_GPSMatrix_50meters_Test.mat');
    elseif strcmp(Dataset,'Qld')
        
        
    elseif strcmp(Dataset,'Oxford')
        Qfol = '/home/n7542704/Datasets/oxford-data/2014-12-10-18-10-50/stereo/left_rect';
        Rfol = '/home/n7542704/Datasets/oxford-data/2014-12-09-13-21-02/stereo/left_rect';
        GT_file = load('/home/n7542704/Datasets/oxford-data/OxfordRobotCar_GPSMatrix_30m.mat');
        %need to re-make the GPS matrix!
    end
else
    if strcmp(Dataset,'Nordland')
        Qfol = 'D:\Windows\Nordland\nordland_summer_cont';
        Rfol = 'D:\Windows\Nordland\nordland_winter_cont';
        GT_file = load('D:\Windows\Nordland\Nordland_GPSMatrix.mat');
    elseif strcmp(Dataset,'Berlin')
        Qfol = 'D:\Windows\Berlin_Kudamm\Query';
        Rfol = 'D:\Windows\Berlin_Kudamm\Reference';
        GT_file = load('D:\Windows\Berlin_Kudamm\Berlin_GPSMatrix_50meters.mat');
    elseif strcmp(Dataset,'Qld')
        
        
    elseif strcmp(Dataset,'Oxford')
        Qfol = 'D:\Windows\oxford-data\2014-12-10-18-10-50\stereo\left_rect';
        Rfol = 'D:\Windows\oxford-data\2014-12-09-13-21-02\stereo\left_rect';
        GT_file = load('D:\Windows\oxford-data\OxfordRobotCar_GPSMatrix_30m.mat');
    end
end
end


