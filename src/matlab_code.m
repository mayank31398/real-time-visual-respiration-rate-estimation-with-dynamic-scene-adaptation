clear
close all
path ='./Data/video.mp4' ;

n=3;

vidObj=VideoReader(path);
N=get(vidObj, 'numberOfFrames');
fs=get(vidObj, 'FrameRate');
N=300;
frame00 = read(vidObj,1);
heig2t = size(frame00,1);
width = size(frame00,2);
display('Select the ROI');
% [~, rect] = imcrop(frame00);
close;


nof=(fs*120);

% rmin0 = rect(2); rmax0 = rect(2)+rect(4);
% cmin0 = rect(1); cmax0 = rect(1)+rect(3);
    
% rowsROI = round(rmin0:rmax0);
% colsROI = round(cmin0:cmax0);

rowsROI = [1: 720];
colsROI = [1: 1280];

tic

off=floor(0*fs*0);
for i=2:n:N-n+1
    i
    V=read(vidObj,[i+off i+n-1+off]); 
    V=double(V(rowsROI,colsROI,:,:));
    V=V(:,:,:,1)/2+V(:,:,:,2)/4 +V(:,:,:,3)/2;
    VIDEO(:,:,floor(i/n)+1) = 0.29*V(:,:,1)+0.59*V(:,:,2)+0.11*V(:,:,3);
end


scale=[120,160];

Lambda=0.8;
MA=25;

Filter=fspecial('gaussian',24,2);

ImageOld=imresize(imfilter(VIDEO(:,:,1),Filter),scale,'bilinear');

RespirationSignal=zeros(size(VIDEO,3),1);

for i = 1:size(VIDEO,3)
    i
    ImageNew=imresize(imfilter(VIDEO(:,:,i),Filter),scale,'bilinear');
    Flow=ComputeFlowField1(ImageNew,ImageOld,0.25);
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
    end
end

t=(1:floor((N-n+1)/n)+1)./(fs/n);

% fs1=fs/n;

% [b,a]=butter(2,[.1/fs1 2.2/fs1]);

% rs=filtfilt(b,a,(RespirationSignal(1:end)));
t=t(1:length(RespirationSignal));
plot(t,RespirationSignal)

RespirationSignal(1:10)

function FlowField=ComputeFlowField1(ImageCurrent,ImageOld,scale)
    [Gradient(:,:,1),Gradient(:,:,2)]=gradient(ImageCurrent);

    GradientMagnitude=sum(Gradient.*Gradient,3);
    GradientMagnitude(GradientMagnitude<9)=inf;
    ImageDiff=ImageOld-ImageCurrent;
    FlowField=bsxfun(@times,(Gradient),((ImageDiff)./(GradientMagnitude)));
    FlowField=imresize(FlowField,scale,'bilinear');
end

