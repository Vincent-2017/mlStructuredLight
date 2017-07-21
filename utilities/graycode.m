function [P,offset] = graycode(width,height)

% % Define height and width of screen.
% width  = 1024;
% height = 768;

% Generate Gray codes for vertical and horizontal stripe patterns.
% The output is a sequence of 2xlog2(width) + 2xlog2(height) + 2 uncompressed images.
% See: http://en.wikipedia.org/wiki/Gray_code
P = cell(2,1); % ����ͼ��
offset = zeros(2,1);
for j = 1:2
   
%    Allocate storage for Gray code stripe pattern. 
%    j��ʾ���� Ϊ������������������ģʽ����洢 
%    Ĭ��NΪ10 ����width  = 1024�� height = 768;
%    ����ͼ���С��768��X1024�� �������С��10��X1024�� ����Ϊ��������ʱ��ÿ������ֻ����һ������
   if j == 1 % 1X1024
      N = ceil(log2(width)); %ceil(x)������ȡ�� 10 ������Ŀ��������λ��
      offset(j) = floor((2^N-width)/2); %floor(x)������ȡ�� 0
   else % 768X1
      N = ceil(log2(height)); % 10 ������Ŀ
      offset(j) = floor((2^N-height)/2); % Ϊ�����߶Գƣ����Ҹ�ȥ��һ�� 128
   end
   % ��ʼ��
   P{j} = zeros(height,width,N,'uint8'); % 768 X 1024 X 10
   
   % Generate N-bit Gray code sequence.
   % ����Nλ�Ķ�ֵ������
   B = zeros(2^N,N,'uint8'); % 1024 X 10 uint8
   B_char = dec2bin(0:2^N-1); % 1024 X 10 char
   % ��ʮ������ 0 ~ 1023 ת����һ���ַ�����ʽ��ʾ�Ķ���������
   % B��B_char ��Ϊ 1024 X 10 �����飬��10λ 0000000000 ~ 1111111111
   for i = 1:N
      B(:,i) = str2num(B_char(:,i));
      % �ַ�ת����
      % B(:,:)
      % 0000000000
      % 0000000001
      % 0000000010
      % ......
      % ......
      % 1111111111
   end
   G = zeros(2^N,N,'uint8'); % 1024 X 10 uint8
   G(:,1) = B(:,1); % ��ֵ��һ��
   for i = 2:N
      G(:,i) = xor(B(:,i-1),B(:,i)); % ������λ���,��ֵ��ת��ɸ�����  % 1024 X 10 uint8
      % G(:,:)  ������
      % 0000000000
      % 0000000001
      % 0000000011
      % ......
      % ......
      % 1111111111
   end
   
   % Store Gray code stripe pattern.
   % �洢����������
   % P{j} = zeros(height,width,N,'uint8');
   if j ==1 
      for i = 1:N % ��������i��λͼ
         P{j}(:,:,i) = repmat(G((1:width)+offset(j),i)',height,1); 
         % ��i=1Ϊ�� ��03.bmp
         % P{j}(:,:,1) = repmat(G((1:width)+offset(j),1)',height,1)=repmat(G((1:width),1)',768,1);
         % G((1:width),1)' 1X1024  �� P{j}(:,:,1) 768X1024 
      end
   else
      for i = 1:N % ��������i��λͼ 
         P{j}(:,:,i) = repmat(G((1:height)+offset(j),i),1,width);
         % ��i=1Ϊ�� ��23.bmp
         % P{j}(:,:,1) = repmat(G((1:height)+offset(j),1),1,width)=repmat(G((1:height)+128,1),1,1024);
         % G((1:height)+128,1)  768X1  �� P{j}(:,:,1) 768X1024
         % ֻȡ�м���129-896,��Ϊ����ͼ���ֻ��768�Ҹ������ǹ������߶ԳƵ�,���ұ�Ե����128 
      end
   end
   
end

% % Decode Gray code stripe patterns. ����
% C = gray2dec(P{1})-offset(1);
% R = gray2dec(P{2})-offset(2);