function [P,offset] = graycode(width,height)

% % Define height and width of screen.
% width  = 1024;
% height = 768;

% Generate Gray codes for vertical and horizontal stripe patterns.
% The output is a sequence of 2xlog2(width) + 2xlog2(height) + 2 uncompressed images.
% See: http://en.wikipedia.org/wiki/Gray_code
P = cell(2,1);
offset = zeros(2,1);
for j = 1:2
   
   % Allocate storage for Gray code stripe pattern. 
   % 为横向和纵向格雷码条纹模式分配存储 
   % 默认N为10 ，因width  = 1024， height = 768;
   if j == 1 
      N = ceil(log2(width)); %ceil(x)：向右取整 10
      offset(j) = floor((2^N-width)/2); %floor(x)：向左取整 0
   else
      N = ceil(log2(height)); % 10
      offset(j) = floor((2^N-height)/2); % 128
   end
   % 赋初值
   P{j} = zeros(height,width,N,'uint8');
   
   % Generate N-bit Gray code sequence.
   % 产生N位的二值码序列
   B = zeros(2^N,N,'uint8');
   B_char = dec2bin(0:2^N-1); 
   %把十进制数 0 ~ 1023 转换成一个字符串形式表示的二进制数。
   % B、B_char 均为 1024 X 10 的数组，有10位 0000000000 ~ 1111111111
   % 编码形式：宽度为1024，每条条纹有10位
   % B_char
   for i = 1:N
      B(:,i) = str2num(B_char(:,i));
      % 字符转数字，得到每一列的二值码（得到1024条纹上对应位数据，共1024位）
      % B(:,:)
   end
   G = zeros(2^N,N,'uint8');
   G(:,1) = B(:,1); % 赋值1024条纹的最高位
   for i = 2:N
      G(:,i) = xor(B(:,i-1),B(:,i)); % 相邻两位异或,二值码转变成格雷码
      % G(:,:) 
   end
   
   % Store Gray code stripe pattern.
   % 存储格雷码条纹
   % P{j} = zeros(height,width,N,'uint8');
   if j ==1 
      for i = 1:N
     %    if(i==1)
     %       offset(j)
     %       G((1:width)+offset(j),1)'
      %      height
      %   end
         P{j}(:,:,i) = repmat(G((1:width)+offset(j),i)',height,1);
      end
   else
      for i = 1:N
      %   if(i==1)
      %      offset(j)
      %      G((1:height)+offset(j),1)'
      %   end
         P{j}(:,:,i) = repmat(G((1:height)+offset(j),i),1,width);
      end
   end
   
end

% % Decode Gray code stripe patterns.
% C = gray2dec(P{1})-offset(1);
% R = gray2dec(P{2})-offset(2);