% 重置matlab环境
clear; clc;

% 加入工具包
addpath('./utilities');

% 设置结构光参数
objName      = 'man';   % 物体名称
seqName      = 'v1';    % 序列名称
seqType      = 'Gray';  % 结构光序列类型
dSampleProj  = 1;       % 采样因子（最小为系统分辨率）
projValue    = 255;     % 格雷码强度
minContrast  = 0.2;     % 最低对比度阈值 (格雷码模式)

% 设置重建参数
dSamplePlot = 100;      % down-sampling rate for Matlab point cloud display
distReject  = Inf;      % rejection distance (for outlier removal)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part I: Project Grey code sequence to recover illumination plane(s).

% 加载系统标定参数
load('./calib/calib_results/calib_cam_proj.mat');
  
% 提示开始扫描
clc; disp('[Reconstruction of Structured Light Sequences]');

% 确定相机的数量和图像分辨率
disp('+ Extracting data set properties...');
D = dir(['./data/',seqType,'/',objName]);
% matlab中使用isdir时为什么要减2才能得到文件数量？
% 因为在我们使用的文件系统中，每个文件夹下都默认含有“.”,“..”两个隐藏的系统文件夹，前者指向该文件夹，后者指向该文件夹的父文件夹，所以要减去2
nCam = nnz([D.isdir])-2;  % number of camera 相机数量 1 ，[D.isdir]中不为0的元素个数为 1 
% disp(['+ nCam = ',int2str(nCam)]);
nBitPlanes = cell(1,nCam); % 照明平面，创建一个空的1x1的cell矩阵,存储不同的数据类型
camDim = cell(1,nCam);
for camIdx = 1:nCam % 相机索引
   dataDir = ['./data/',seqType,'/',objName,'/v',int2str(camIdx),'/'];
   % disp(dataDir);
   nBitPlanes{camIdx} = ((length(dir(dataDir))-2)-2)/4; % 10 
   % length(dir(dataDir)) = 44
   % dir('G:\Matlab')列出指定目录下所有子文件夹和文件
   % 图片数目 2xlog2(width) + 2xlog2(height) + 2  = 42 故 nBitPlanes{camIdx} = 10
   I = imread([dataDir,'01.bmp']);
   camDim{camIdx} = [size(I,1) size(I,2)];
end
width = camDim{1}(2); % 摄像机捕捉的图像分辨率 1200 X 1600
height = camDim{1}(1);
disp(['+ The large of image is ',int2str(height),' X ',int2str(width)]);

% Generate vertical and horizontal Gray code stripe patterns.
% Note: P{j} contains the Gray code patterns for "orientation" j. 方向j
%       offset(j) is the integer column/row offset for P{j}.
%       I{j,i} are the OpenGL textures corresponding to bit i of P{j}.
%       J{j,i} are the OpenGL textures of the inverse of I{j,i}.
disp('+ Regenerating structured light sequence...');
if strcmp(seqType,'Gray') %strcmp是用于做字符串比较的函数
   [P,offset] = graycode(1024/dSampleProj,768/dSampleProj);  % 投影图案的大小 1024 X 768
else
   [P,offset] = bincode(1024/dSampleProj,768/dSampleProj);
end

% 加载拍摄的结构光序列
disp('+ Loading data set...');
for camIdx = 1:nCam %  nCam = 1    
   dataDir = ['./data/',seqType,'/',objName,'/v',int2str(camIdx),'/']; % ./data/Gray/man/v1/
   if ~exist(dataDir,'dir')
      error(['Sequence ',objName,'_',seqName,'_',seqType,' is not available!']);
   end
   % 前两张全白和全黑照片
   T{1}{camIdx} = imread([dataDir,num2str(1,'%0.02d'),'.bmp']); % 全白
   T{2}{camIdx} = imread([dataDir,num2str(2,'%0.02d'),'.bmp']); % 全黑
   % 剩余的行列照片
   frameIdx = 3;
   for j = 1:2
      for i = 1:nBitPlanes{camIdx}        % nBitPlanes{camIdx} = 10 
         A{j,i}{camIdx} = imread([dataDir,num2str(frameIdx,'%0.02d'),'.bmp']);
         frameIdx = frameIdx + 1;
         B{j,i}{camIdx} = imread([dataDir,num2str(frameIdx,'%0.02d'),'.bmp']);
         frameIdx = frameIdx + 1;
         % A、B 对应的图片互补
         % A{1，1} = 03.bmp B{1，1} = 04.bmp 
         % A{1，2} = 05.bmp B{1，2} = 06.bmp
         % ....
         % A{1，10} = 21.bmp B{1，2} = 22.bmp
         %
         % A{2，1} = 23.bmp B{2，1} = 24.bmp 
         % A{2，2} = 25.bmp B{2，2} = 26.bmp
         % ....
         % A{2，10} = 41.bmp B{2，2} = 42.bmp
      end
   end
end

% Estimate column/row label for each pixel (i.e., decode Gray codes).
% Note: G{j,k} is the estimated Gray code for "orientation" j and camera k.
%       j 定义方向、k定义相机
%       D{j,k} is the integer column/row estimate. 估计
%       M{j,k} is the per-pixel mask (i.e., pixels with enough contrast). 掩码 
disp('+ Recovering projector rows/columns from structured light sequence...');
G = cell(size(A,1),nCam); % 2x10x1
D = cell(size(A,1),nCam);
M = cell(size(A,1),nCam);
C = inv([1.0 0.956 0.621; 1.0 -0.272 -0.647; 1.0 -1.106 1.703]); % 矩阵求逆  RGB颜色权重
C = C(1,:)';
for k = 1:nCam
   for j = 1:size(A,1) % 数组A的一维方向长度 2 方向
      % 数组T{1}{1}（全白图像）的一、二维方向长度 ， 数组A{2,10}的二维方向长度
      G{j,k} = zeros(size(T{1}{1},1),size(T{1}{1},2),size(A,2),'uint8'); % 1200 x 1600 x 10 uint8
      M{j,k} = false(size(T{1}{1},1),size(T{1}{1},2)); % 逻辑0矩阵 1200 x 1600 logical
      for i = 1:size(A,2) % 10
         % Convert image pair to grayscale.
         %grayA = rgb2gray(im2double(A{j,i}{k}));
         %grayB = rgb2gray(im2double(B{j,i}{k}));
         % imlincomb 线性组合
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
         
         % Eliminate all pixels that do not exceed contrast threshold.
         % 超过对比度阈值的所有像素赋值为true
         M{j,k}(abs(grayA-grayB) > 255*minContrast) = true;
         
         % Estimate current bit of Gray code from image pair. 
         % 估计 图像对 当前的格雷码
         bitPlane = zeros(size(T{1}{1},1),size(T{1}{1},2),'uint8');
         % 1200 x 1600 uint8
         % temp = grayA(:,:) >= grayB(:,:); % 1200 x 1600 logical
         bitPlane(grayA(:,:) >= grayB(:,:)) = 1; 
         G{j,k}(:,:,i) = bitPlane;   
      end
      if strcmp(seqType,'Gray')
         D{j,k} = gray2dec(G{j,k})-offset(j);
      else
         D{j,k} = bin2dec(G{j,k})-offset(j);
      end
      D{j,k}(~M{j,k}) = NaN;
   end
end
%clear A B G grayA grayB bitPlane;

% Eliminate invalid column/row estimates. 消除无效的列/行估计
% Note: This will exclude pixels if either the column or row is missing.
%       D{j,k} is the column/row for "orientation" j and camera k.
%       mask{k} is the overal per-pixel mask for camera k.
mask = cell(1,nCam);
for k = 1:nCam
   mask{k} = M{1,k};
   for j = 1:size(D,1)
      if j == 1
         D{j,k}(D{j,k} > width) = NaN;
      else
         D{j,k}(D{j,k} > height) = NaN;
      end
      D{j,k}(D{j,k} < 1) = NaN;
      for i = 1:size(D,1)
         D{j,k}(~M{i,k}) = NaN;
         mask{k} =  mask{k} & M{i,k};
      end
   end
end

% Display recovered projector column/row.
figure(1); clf;
imagesc(D{1,1}); axis image; colormap(jet(256));
title('Recovered Projector Column Indices'); drawnow;
figure(2); clf;
imagesc(D{2,1}); axis image; colormap(jet(256));
title('Recovered Projector Row Indices'); drawnow;
figure(3); clf;
imagesc(T{1}{1}); axis image; colormap(jet(256));
title('Reference Image for Texture Mapping'); drawnow;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part II: Reconstruct surface using line-plane intersection.

% Reconstruct 3D points using intersection with illumination plane(s).
% Note: Reconstructs from all cameras in the first camera coordinate system.
vertices = cell(1,length(Nc));
colors   = cell(1,length(Nc));
disp('+ Reconstructing 3D points...');
for i = 1:length(Nc)
   idx       = find(~isnan(D{1,i}) & ~isnan(D{2,i}));
   [row,col] = ind2sub(size(D{1,i}),idx);
   npts      = length(idx);
   colors{i} = 0.65*ones(npts,3);
   Rc        = im2double(T{1}{i}(:,:,1));
   Gc        = im2double(T{1}{i}(:,:,2));
   Bc        = im2double(T{1}{i}(:,:,3));
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
