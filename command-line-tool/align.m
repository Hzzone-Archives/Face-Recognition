function [cropImg] = align(imgpath, facial5points)
imgSize = [112, 96];
coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041];

image = imread(imgpath);
% facial5points = [347.32406616, 535.37561035, 415.44692993 , 345.96670532, 498.96899414;406.24920654,435.16409302, 539.11853027,592.89459229 , 617.25360107];
facial5points = [ 108.01406097 , 146.49569702,  120.82705688 ,  99.81839752,  128.7023468;110.5831604   ,119.56469727 , 126.31167603 , 148.01054382,  155.75180054];
Tfm =  cp2tform(facial5points', coord5points', 'similarity');
cropImg = imtransform(image, Tfm, 'XData', [1 imgSize(2)],...
                                  'YData', [1 imgSize(1)], 'Size', imgSize);
% imshow(cropImg)
% imwrite(cropImg, "1.jpg")
end