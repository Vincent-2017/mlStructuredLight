% ����matlab����
clear; clc;close all ;

% ���빤�߰�
addpath('./utilities');

% ���ýṹ�����
objName      = 'man';   % ��������
seqName      = 'v1';    % ��������
seqType      = 'Gray';  % �ṹ����������
dSampleProj  = 1;       % �������ӣ���СΪϵͳ�ֱ��ʣ�
projValue    = 255;     % ������ǿ��
minContrast  = 0.2;     % ��ͶԱȶ���ֵ (������ģʽ)

% �����ؽ�����
dSamplePlot = 100;      % down-sampling rate for Matlab point cloud display
distReject  = Inf;      % rejection distance (for outlier removal)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part I: Project Grey code sequence to recover illumination plane(s).

% ����ϵͳ�궨����
load('./calib/calib_results/calib_cam_proj.mat');
 
% ��ʾ��ʼɨ��
clc; disp('[Reconstruction of Structured Light Sequences]');

% ȷ�������������ͼ��ֱ���
disp('+ Extracting data set properties...');
D = dir(['./data/',seqType,'/',objName]);
% ÿ���ļ����¶�Ĭ�Ϻ��С�.��,��..���������ص�ϵͳ�ļ��У�ǰ��ָ����ļ��У�����ָ����ļ��еĸ��ļ��У�����Ҫ��ȥ2
nCam = nnz([D.isdir])-2;  % ������� 1 ��[D.isdir]�в�Ϊ0��Ԫ�ظ���Ϊ 1 
nBitPlanes = cell(1,nCam); % nλ��
camDim = cell(1,nCam);
for camIdx = 1:nCam % �������
   dataDir = ['./data/',seqType,'/',objName,'/v',int2str(camIdx),'/']; % ./data/Gray/man/v1/
   nBitPlanes{camIdx} = ((length(dir(dataDir))-2)-2)/4; % ��ȥ���������ļ��У���ȥ��ȫ��ȫ�ף����к��������������2
   % length(dir(dataDir)) = 44
   % dir('G:\Matlab')�г�ָ��Ŀ¼���������ļ��к��ļ�
   % ͼƬ��Ŀ 2xlog2(width) + 2xlog2(height) + 2  = 42 �� nBitPlanes{camIdx} = 10
   I = imread([dataDir,'01.bmp']);
   camDim{camIdx} = [size(I,1) size(I,2)]; % 1200 X 1600
end
% �������׽��ͼ��ֱ��� 1200 X 1600
width = camDim{1}(2); 
height = camDim{1}(1);
disp(['+ The large of image is ',int2str(height),' X ',int2str(width)]);

% Generate vertical and horizontal Gray code stripe patterns.
% Note: P{j} contains the Gray code patterns for "orientation" j. ����j
%       offset(j) is the integer column/row offset for P{j}.
%       I{j,i} are the OpenGL textures corresponding to bit i of P{j}.
%       J{j,i} are the OpenGL textures of the inverse of I{j,i}.
disp('+ Regenerating structured light sequence...');
if strcmp(seqType,'Gray') %strcmp���������ַ����Ƚϵĺ���
   [P,offset] = graycode(1024/dSampleProj,768/dSampleProj);  % ����ͶӰͼ��,��С 1024 X 768
else
   [P,offset] = bincode(1024/dSampleProj,768/dSampleProj);
end

% ��������Ľṹ������
disp('+ Loading data set...');
for camIdx = 1:nCam %  nCam = 1    
   dataDir = ['./data/',seqType,'/',objName,'/v',int2str(camIdx),'/']; % ./data/Gray/man/v1/
   if ~exist(dataDir,'dir')
      error(['Sequence ',objName,'_',seqName,'_',seqType,' is not available!']);
   end
   % ǰ����ȫ�׺�ȫ����Ƭ
   T{1}{camIdx} = imread([dataDir,num2str(1,'%0.02d'),'.bmp']); % ȫ�� 01.bmp
   T{2}{camIdx} = imread([dataDir,num2str(2,'%0.02d'),'.bmp']); % ȫ�� 02.bmp
   % ʣ���������Ƭ
   frameIdx = 3;
   for j = 1:2 % j ����
      for i = 1:nBitPlanes{camIdx}        % nBitPlanes{camIdx} = 10 λ�� 
         A{j,i}{camIdx} = imread([dataDir,num2str(frameIdx,'%0.02d'),'.bmp']);
         frameIdx = frameIdx + 1;
         B{j,i}{camIdx} = imread([dataDir,num2str(frameIdx,'%0.02d'),'.bmp']);
         frameIdx = frameIdx + 1;
         % A��B ��Ӧ��ͼƬ���� ���10��ͼ���
         % A{1��1} = 03.bmp B{1��1} = 04.bmp 
         % A{1��2} = 05.bmp B{1��2} = 06.bmp
         % ....
         % A{1��10} = 21.bmp B{1��2} = 22.bmp
         %
         % A{2��1} = 23.bmp B{2��1} = 24.bmp 
         % A{2��2} = 25.bmp B{2��2} = 26.bmp
         % ....
         % A{2��10} = 41.bmp B{2��2} = 42.bmp
      end
   end
end

% Estimate column/row label for each pixel (i.e., decode Gray codes).
% Note: G{j,k} is the estimated Gray code for "orientation" j and camera k.
%       j ���巽��k�������
%       D{j,k} is the integer column/row estimate. ����
%       M{j,k} is the per-pixel mask (i.e., pixels with enough contrast). ���� 
disp('+ Recovering projector rows/columns from structured light sequence...');
G = cell(size(A,1),nCam); % 2x10x1
D = cell(size(A,1),nCam);
M = cell(size(A,1),nCam);
C = inv([1.0 0.956 0.621; 1.0 -0.272 -0.647; 1.0 -1.106 1.703]); % ��������  RGB��ɫȨ��
C = C(1,:)';
for k = 1:nCam
   for j = 1:size(A,1) % ����A��һά���򳤶� 2 ����
      % ����T{1}{1}��ȫ��ͼ�񣩵�һ����ά���򳤶� �� ����A{2,10}�Ķ�ά���򳤶�
      G{j,k} = zeros(size(T{1}{1},1),size(T{1}{1},2),size(A,2),'uint8'); % 1200 x 1600 x 10 uint8
      M{j,k} = false(size(T{1}{1},1),size(T{1}{1},2)); % �߼�0���� 1200 x 1600 logical
      for i = 1:size(A,2) % 10λ��
         % Convert image pair to grayscale. ת��ͼ��Ե�������
         % grayA = rgb2gray(im2double(A{j,i}{k}));
         % grayB = rgb2gray(im2double(B{j,i}{k}));
         % imlincomb �������
         % C(1) = 0.2989
         % C(2) = 0.5870
         % C(3) = 0.1140   
         % A{j,i}{k}(:,:,1) R
         % A{j,i}{k}(:,:,2) G
         % A{j,i}{k}(:,:,3) B
         grayA = imlincomb(C(1),A{j,i}{k}(:,:,1),...
                           C(2),A{j,i}{k}(:,:,2),...
                           C(3),A{j,i}{k}(:,:,3),'double');  % 1200 x 1600 double        
         grayB = imlincomb(C(1),B{j,i}{k}(:,:,1),...
                           C(2),B{j,i}{k}(:,:,2),...
                           C(3),B{j,i}{k}(:,:,3),'double');

         % imshow(grayA);
         % imshow(grayB);
         % image = grayA-grayB;
         % absimage = abs(grayA-grayB);
         % imshow(grayA-grayB);
         % imshow(abs(grayA-grayB));
         % Eliminate all pixels that do not exceed contrast threshold.
         % �����Աȶ���ֵ51��λ�ø�ֵΪtrue
         M{j,k}(abs(grayA-grayB) > 255*minContrast) = true; % {[1200x1600 logical' char(10) ']}   
         % Estimate current bit of Gray code from image pair. 
         % ���� ��ǰͼ��� �ĸ�����λ��
         bitPlane = zeros(size(T{1}{1},1),size(T{1}{1},2),'uint8');  % 1200 x 1600 uint8
         % temp = grayA(:,:) >= grayB(:,:); % 1200 x 1600 logical
         bitPlane(grayA(:,:) >= grayB(:,:)) = 1; % �� 03.bmp 05.bmp ... 21.bmp ��A��Ϊ׼
         % imshow(bitPlane);
         G{j,k}(:,:,i) = bitPlane;  % �õ���i��λ�棬��10λ�� 
      end
      if strcmp(seqType,'Gray')
         D{j,k} = gray2dec(G{j,k})-offset(j); % ���� 1200X1600X10 ������ ת1200X1600 ʮ����
      else
         D{j,k} = bin2dec(G{j,k})-offset(j);
      end
      D{j,k}(~M{j,k}) = NaN; % �������л����е�һ����Ҫ���ȫ����ֵNaN
   end
end
%clear A B G grayA grayB bitPlane;

% Eliminate invalid column/row estimates. ������Ч����/�й��ƣ���������ͼ��1024X768��С��
% Note: This will exclude pixels if either the column or row is missing.
%       D{j,k} is the column/row for "orientation" j and camera k.
%       mask{k} is the overal per-pixel mask for camera k.
mask = cell(1,nCam);
for k = 1:nCam % k=1
   mask{k} = M{1,k}; % mask��ʼ��Ϊ��һ���룬��ʱM{j,k}�����з�������һ��λ�������
   for j = 1:size(D,1)
      if j == 1
         D{j,k}(D{j,k} > width) = NaN; % �з���Ϳ�ȱȽ�
      else
         D{j,k}(D{j,k} > height) = NaN; % �з���͸߶����Ƚ�
      end
      D{j,k}(D{j,k} < 1) = NaN; % ʮ��������������С��1
      for i = 1:size(D,1) % 2
         D{j,k}(~M{i,k}) = NaN; % ������������������Ҫ���ȫ����ֵNaN
         mask{k} =  mask{k} & M{i,k}; % mask Ϊ˫����
      end
   end
end

% Display recovered projector column/row.
figure(1); clf;
imagesc(D{1,1}); axis image; colormap(jet(256)); % ��
title('Recovered Projector Column Indices'); drawnow;
figure(2); clf;
imagesc(D{2,1}); axis image; colormap(jet(256)); % ��
title('Recovered Projector Row Indices'); drawnow;
figure(3); clf;
imagesc(T{1}{1}); axis image; colormap(jet(256)); % ȫ��ͼ��
title('Reference Image for Texture Mapping'); drawnow;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part II: Reconstruct surface using line-plane intersection.

% Reconstruct 3D points using intersection with illumination plane(s).
% Note: Reconstructs from all cameras in the first camera coordinate system.
vertices = cell(1,length(Nc));
colors   = cell(1,length(Nc));
disp('+ Reconstructing 3D points...');
for i = 1:length(Nc)
   idx       = find(~isnan(D{1,i}) & ~isnan(D{2,i})); % �ҵ�����������
   [row,col] = ind2sub(size(D{1,i}),idx); % ����ת�����±�
   npts      = length(idx);
   colors{i} = 0.65*ones(npts,3);
   Rc        = im2double(T{1}{i}(:,:,1));
   Gc        = im2double(T{1}{i}(:,:,2));
   Bc        = im2double(T{1}{i}(:,:,3)); % ��ɫ
   vV = intersectLineWithPlane(repmat(Oc{i},1,npts),Nc{i}(:,idx),wPlaneCol(D{1,i}(idx),:)');
   vH = intersectLineWithPlane(repmat(Oc{i},1,npts),Nc{i}(:,idx),wPlaneRow(D{2,i}(idx),:)');
   vertices{i} = vV';
   rejectIdx = find(sqrt(sum((vV-vH).^2)) > distReject);
   vertices{i}(rejectIdx,1) = NaN;
   vertices{i}(rejectIdx,2) = NaN;
   vertices{i}(rejectIdx,3) = NaN;
   colors{i}(:,1) = Rc(idx);
   colors{i}(:,2) = Gc(idx);
   colors{i}(:,3) = Bc(idx);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part III: Display reconstruction results and export VRML model.

% Display status.
disp('+ Displaying results and exporting VRML model...');

% Display project/camera calibration results.
procamCalibDisplay;

% Display the recovered 3D point cloud (with per-vertex color).
% Note: Convert to indexed color map for use with FSCATTER3.
for i = 1:length(Nc)
   C = reshape(colors{i},[size(colors{i},1) 1 size(colors{i},2)]);
   [C,cmap] = rgb2ind(C,256);
   hold on;
      fscatter3(vertices{i}(1:dSamplePlot:end,1),...
                vertices{i}(1:dSamplePlot:end,3),...
               -vertices{i}(1:dSamplePlot:end,2),...
                double(C(1:dSamplePlot:end)),cmap);
   hold off;
   axis tight; drawnow;
end

% Export colored point cloud as a VRML file.
% Note: Interchange x and y coordinates for j3DPGP.
clear idx; mergedVertices = []; mergedColors = [];
for i = 1:length(Nc)
   idx{i} = find(~isnan(vertices{i}(:,1)));
   vertices{i}(:,2) = -vertices{i}(:,2);
   vrmlPoints(['./data/',seqType,'/',objName,'/v',int2str(i),'.wrl'],...
      vertices{i}(idx{i},[1 2 3]),colors{i}(idx{i},:));
   mergedVertices = [mergedVertices; vertices{i}(idx{i},[1 2 3])];
   mergedColors = [mergedColors; colors{i}(idx{i},:)];
end
if length(Nc) > 1
   vrmlPoints(['./data/',seqType,'/',objName,'/merged.wrl'],...
      mergedVertices,mergedColors);
end
disp(' ');
