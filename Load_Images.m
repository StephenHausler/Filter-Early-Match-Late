function [filesD,filesQ,dSize,qSize] = Load_Images(Query_folder,Ref_folder,...
    imStartR,imStartQ,frameSkip,imEndR,imEndQ)

fR_temp = dir(Ref_folder);
for i = 1:length(fR_temp)
    name = fR_temp(i).name;
    patns = {'jpeg','jpg','png'};
    for j = 1:length(patns)  %assuming same filetype across folder
        k = strfind(name,patns{j});
        if k %exists
            file_type = fR_temp(i).name(k:end);
            break
        end
    end
    if k
        break
    end
end
Ref_file_type = strcat('*',file_type);
fR = dir(fullfile(Ref_folder,Ref_file_type));
Imcounter_R = imStartR;
fR2 = struct2cell(fR);
tmpFilesR = sort_nat(fR2(1,:));
i = 1;
if nargin == 3
    while(((Imcounter_R+1) <= length(tmpFilesR)) && ((Imcounter_R+1) <= imEndR))
        filenamesR{i} = tmpFilesR(Imcounter_R+1);
        Imcounter_R = Imcounter_R + frameSkip;
        i = i + 1;
    end
else
    while((Imcounter_R+1) <= length(tmpFilesR))
        filenamesR{i} = tmpFilesR(Imcounter_R+1);
        Imcounter_R = Imcounter_R + frameSkip;
        i = i + 1;
    end
end
%define struct member ConfigObj.fR
filesD.fD = filenamesR;
filesD.path = fR(1).folder;

fQ_temp = dir(Query_folder);
for i = 1:length(fQ_temp)
    name = fQ_temp(i).name;
    patns = {'jpeg','jpg','png'};
    for j = 1:length(patns)  %assuming same filetype across folder
        k = strfind(name,patns{j});
        if k %exists
            file_type = fQ_temp(i).name(k:end);
            break
        end
    end
    if k
        break
    end
end
Query_file_type = strcat('*',file_type);
fQ = dir(fullfile(Query_folder,Query_file_type));
Imcounter_Q = imStartQ;
fQ2 = struct2cell(fQ);
tmpFilesQ = sort_nat(fQ2(1,:));
i = 1;
if nargin == 3
    while(((Imcounter_Q+1) <= length(tmpFilesQ)) && ((Imcounter_Q+1) <= imEndQ))
        filenamesQ{i} = tmpFilesQ(Imcounter_Q+1);
        Imcounter_Q = Imcounter_Q + frameSkip;
        i = i + 1;
    end
else
    while((Imcounter_Q+1) <= length(tmpFilesQ))
        filenamesQ{i} = tmpFilesQ(Imcounter_Q+1);
        Imcounter_Q = Imcounter_Q + frameSkip;
        i = i + 1;
    end
end
%define struct member ConfigObj.fQ
filesQ.fQ = filenamesQ;
filesQ.path = fQ(1).folder;

%set size of query and template image sets
dSize = length(filenamesR);
qSize = length(filenamesQ);
end

