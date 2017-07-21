function [P,offset] = graycode(width,height)

% % Define height and width of screen.
% width  = 1024;
% height = 768;

% Generate Gray codes for vertical and horizontal stripe patterns.
% The output is a sequence of 2xlog2(width) + 2xlog2(height) + 2 uncompressed images.
% See: http://en.wikipedia.org/wiki/Gray_code
P = cell(2,1); % 编码图案
offset = zeros(2,1);
for j = 1:2
   
%    Allocate storage for Gray code stripe pattern. 
%    j表示方向 为横向和纵向格雷码条纹模式分配存储 
%    默认N为10 ，因width  = 1024， height = 768;
%    编码图像大小：768行X1024列 格雷码大小：10行X1024列 即当为竖向条纹时，每个条纹只包含一个像素
   if j == 1 % 1X1024
      N = ceil(log2(width)); %ceil(x)：向右取整 10 序列数目，即编码位数
      offset(j) = floor((2^N-width)/2); %floor(x)：向左取整 0
   else % 768X1
      N = ceil(log2(height)); % 10 序列数目
      offset(j) = floor((2^N-height)/2); % 为了中线对称，左右各去掉一半 128
   end
   % 初始化
   P{j} = zeros(height,width,N,'uint8'); % 768 X 1024 X 10
   
   % Generate N-bit Gray code sequence.
   % 产生N位的二值码序列
   B = zeros(2^N,N,'uint8'); % 1024 X 10 uint8
   B_char = dec2bin(0:2^N-1); % 1024 X 10 char
   % 把十进制数 0 ~ 1023 转换成一个字符串形式表示的二进制数。
   % B、B_char 均为 1024 X 10 的数组，有10位 0000000000 ~ 1111111111
   for i = 1:N
      B(:,i) = str2num(B_char(:,i));
      % 字符转数字
      % B(:,:)
      % 0000000000
      % 0000000001
      % 0000000010
      % ......
      % ......
      % 1111111111
   end
   G = zeros(2^N,N,'uint8'); % 1024 X 10 uint8
   G(:,1) = B(:,1); % 赋值第一列
   for i = 2:N
      G(:,i) = xor(B(:,i-1),B(:,i)); % 相邻两位异或,二值码转变成格雷码  % 1024 X 10 uint8
      % G(:,:)  格雷码
      % 0000000000
      % 0000000001
      % 0000000011
      % ......
      % ......
      % 1111111111
   end
   
   % Store Gray code stripe pattern.
   % 存储格雷码条纹
   % P{j} = zeros(height,width,N,'uint8');
   if j ==1 
      for i = 1:N % 列向编码第i幅位图
         P{j}(:,:,i) = repmat(G((1:width)+offset(j),i)',height,1); 
         % 以i=1为例 如03.bmp
         % P{j}(:,:,1) = repmat(G((1:width)+offset(j),1)',height,1)=repmat(G((1:width),1)',768,1);
         % G((1:width),1)' 1X1024  故 P{j}(:,:,1) 768X1024 
      end
   else
      for i = 1:N % 行向编码第i幅位图 
         P{j}(:,:,i) = repmat(G((1:height)+offset(j),i),1,width);
         % 以i=1为例 如23.bmp
         % P{j}(:,:,1) = repmat(G((1:height)+offset(j),1),1,width)=repmat(G((1:height)+128,1),1,1024);
         % G((1:height)+128,1)  768X1  故 P{j}(:,:,1) 768X1024
         % 只取中间行129-896,因为编码图像高只有768且格雷码是关于中线对称的,左右边缘各差128 
      end
   end
   
end

% % Decode Gray code stripe patterns. 解码
% C = gray2dec(P{1})-offset(1);
% R = gray2dec(P{2})-offset(2);