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
   % Ϊ������������������ģʽ����洢 
   % Ĭ��NΪ10 ����width  = 1024�� height = 768;
   if j == 1 
      N = ceil(log2(width)); %ceil(x)������ȡ�� 10
      offset(j) = floor((2^N-width)/2); %floor(x)������ȡ�� 0
   else
      N = ceil(log2(height)); % 10
      offset(j) = floor((2^N-height)/2); % 128
   end
   % ����ֵ
   P{j} = zeros(height,width,N,'uint8');
   
   % Generate N-bit Gray code sequence.
   % ����Nλ�Ķ�ֵ������
   B = zeros(2^N,N,'uint8');
   B_char = dec2bin(0:2^N-1); 
   %��ʮ������ 0 ~ 1023 ת����һ���ַ�����ʽ��ʾ�Ķ���������
   % B��B_char ��Ϊ 1024 X 10 �����飬��10λ 0000000000 ~ 1111111111
   % ������ʽ�����Ϊ1024��ÿ��������10λ
   % B_char
   for i = 1:N
      B(:,i) = str2num(B_char(:,i));
      % �ַ�ת���֣��õ�ÿһ�еĶ�ֵ�루�õ�1024�����϶�Ӧλ���ݣ���1024λ��
      % B(:,:)
   end
   G = zeros(2^N,N,'uint8');
   G(:,1) = B(:,1); % ��ֵ1024���Ƶ����λ
   for i = 2:N
      G(:,i) = xor(B(:,i-1),B(:,i)); % ������λ���,��ֵ��ת��ɸ�����
      % G(:,:) 
   end
   
   % Store Gray code stripe pattern.
   % �洢����������
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