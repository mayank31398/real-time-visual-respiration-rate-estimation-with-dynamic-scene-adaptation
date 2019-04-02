clear
close all
path ='/Users/prathoshap/Documents/RespSpeech/s36.mp4' ;

n=3;

 vidObj=VideoReader(path);
     N=get(vidObj, 'numberOfFrames');
    fs=get(vidObj, 'FrameRate');
 N=3000;
    frame00 = read(vidObj,1);
heig2t = size(frame00,1);
width = size(frame00,2);
display('Select the ROI');
[~, rect] = imcrop(frame00); 
close 


nof=(fs*120);

rmin0 = rect(2); rmax0 = rect(2)+rect(4);
cmin0 = rect(1); cmax0 = rect(1)+rect(3);
    
rowsROI = round(rmin0:rmax0);
colsROI = round(cmin0:cmax0);
%     

tic

off=floor(0*fs*0);
 for i=1:n:N-n+1 
       
     
    V=read(vidObj,[i+off i+n-1+off]); 
    
    i
  
%    V=read(vidObj,[i i+2]); 
       
%   V=read(vidObj,i); 
  V=double(V(rowsROI,colsROI,:,:));
%    V=double(V(:,:,:,:));
%    V=double(V(u:d,l:r,:,:));
%  V=double(V(:,:,:,:));
         
    V=V(:,:,:,1)/2+V(:,:,:,2)/4 +V(:,:,:,3)/2;


  %V=V(:,:,:,1)/8+V(:,:,:,2)*3/8+V(:,:,:,3)*3/8+V(:,:,:,4)/8;

 VIDEO(:,:,floor(i/n)+1) = 0.29*V(:,:,1)+0.59*V(:,:,2)+0.11*V(:,:,3);
      
 end


scale=[120,160];

% scale = 1; 

Lambda=0.8;
MA=25;

Filter=fspecial('gaussian',24,2);

% figure('position',[50 50 1425 725])

ImageOld=imresize(imfilter(VIDEO(:,:,1),Filter),scale,'bilinear');

RespirationSignal=zeros(size(VIDEO,3),1);

for i = 1:size(VIDEO,3)
    ImageNew=imresize(imfilter(VIDEO(:,:,i),Filter),scale,'bilinear');
    
    Flow=ComputeFlowField1(ImageNew,ImageOld,0.25);
%     
%      quiver(Flow(:,:,1),Flow(:,:,2));
% %     plot(Flow(:,:,1))
%      close
    ImageOld=ImageNew;
mf(i)=max(max(max(Flow)));
    
    if(i==1)
        TFLOW=Flow;
        DFLOW=Flow;
    else
        TFLOW=Lambda*TFLOW+Flow;
        mag=sqrt(sum(sum(sum(TFLOW.*TFLOW))));
		if(mag>MA)
            TFLOW=TFLOW*MA/mag;
        end
         
		if(sum(sum(sum(DFLOW.*Flow)))>0)
			DFLOW=Lambda*DFLOW+Flow;
        else
            DFLOW=Lambda*DFLOW-Flow;
        end
    
        mag=sqrt(sum(sum(sum(DFLOW.*DFLOW))));
        if(mag>MA)
            DFLOW=DFLOW*MA/mag;
            mag=MA;
        end
        RespirationVelocity=sum(sum(sum(TFLOW.*DFLOW)))/mag;
        
        if ~ isnan(RespirationVelocity)
        RespirationSignal(i)=0.9*RespirationSignal(i-1)+RespirationVelocity;
        else
        RespirationSignal(i)=0;
        end

		 
    
        % Display
%         subplot(1,2,1);imshow(uint8(VIDEO(:,:,i)));title(['Sub: ',num2str(sub),' Exp: ',num2str(exp),' Time: ',num2str(i/10),' s']);
%          plot(RespirationSignal(max(i-100+1,1):i),'b');hold on; hold off;
% %         plot(5*Pneumograph(max(i-100+1,1):i),'r');hold off;
%         axis([0,100,-25,25]);
%         drawnow;  
    end
end

t=(1:floor((N-n+1)/n)+1)./(fs/n);

fs1=fs/n;

[b,a]=butter(2,[.1/fs1 2.2/fs1]);


rs=filtfilt(b,a,(RespirationSignal(1:end)));
t=t(1:length(RespirationSignal));
% t=t./60;
% subplot(211)
   plot(t,rs)
%    subplot(212)
%    f=((1:512).*fs)./512;
%    spec=10*log10(abs(fft(rs.*hanning(length(rs)),512)));
%    plot(f(1:256),spec(1:256))
% g=1;





% for i=1:length(mixedsig)-batchLength*fs-1
%     
%     
%         
%         s1 = mixedsig(i:i+floor((batchLength)*fs));
%         detrended_s1 = detrend2(s1',15);                                      % detrending s1
%         filtered_s1 = detrended_s1; %filtfilt(b,a,detrended_s1);                               % bandpass filtering s1
%         autocorrelated_s1 = xcorr(filtered_s1,'unbiased');                      % autocorrelated s1
%         VPGsignal = autocorrelated_s1(1:(end+1)/2);
%         VPGsignal = (VPGsignal - mean(VPGsignal))/std(VPGsignal);               % Normalization
%         
%        
%         
%         [HR, SNR_area] = SNR( smooth(VPGsignal), fs, 15, 0, 0 );
%         
% %        
% %         
%         RR_all = [RR_all; HR];
%         %SNR_all = [SNR_all; SNR_area];
%         
%          close all;
%         
%       
%     
% end 


function FlowField=ComputeFlowField1(ImageCurrent,ImageOld,scale)
% Computes FlowField from the current and the previous frames
% INPUT:  ImageCurrent = Current video frame
%         ImageOld = Previous frame
% OUTPUT: FlowField = Computed Flow Field

% Equations 1,2 and 3 in the paper. 
[Gradient(:,:,1),Gradient(:,:,2)]=gradient(ImageCurrent);

GradientMagnitude=sum(Gradient.*Gradient,3);
GradientMagnitude(GradientMagnitude<9)=inf;
ImageDiff=ImageOld-ImageCurrent;
% ImageDiff(abs(ImageDiff)<0.25)=0;
FlowField=bsxfun(@times,(Gradient),((ImageDiff)./(GradientMagnitude)));
FlowField=imresize(FlowField,scale,'bilinear');
end

