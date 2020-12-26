% IET MATLAB WORKSHOP 2020
% Author : Ashwin De Silva

%% Image Processing Basic Concepts

clear all;
close all;

%% reading an image

im = imread('original.png');
imshow(im);

%% getting the image size

% disp('Image Size : ');
% disp(size(im));

%% converting to grayscale

% im_gray = rgb2gray(im);
% figure; 
% subplot(121); imshow(im);
% subplot(122); imshow(im_gray);

%% rotating an image

% im_90 = imrotate(im, 90);
% figure; 
% subplot(121); imshow(im);
% subplot(122); imshow(im_90);

%% resizing an image

% im_resized = imresize(im, 2); % rescale by a factor of 2
% figure; imshow(im);
% figure; imshow(im_resized);

%% cropping an image

% J = imcrop(im); % double click on the selected rectangle to crop
% figure; 
% subplot(121); imshow(im);
% subplot(122); imshow(J);

%% recovering the R, G and B planes

% figure; imshow(im);
% 
% im_r = im(:, :, 1);
% im_g = im(:, :, 2);
% im_b = im(:, :, 3);
% 
% figure;
% subplot(131); imshow(im_r); title('Red Channel');
% subplot(132); imshow(im_g); title('Green Channel');
% subplot(133); imshow(im_b); title('Blue Channel');
% 
% im(:, :, 1) = 0;
% im(:, :, 3) = 0;
% 
% figure; imshow(im);

%% Brightness
% 
% B = 40;
% im_bright = im - 40;
% 
% im_bright(im_bright < 0) = 0;
% 
% figure;
% subplot(121); imshow(im);
% subplot(122); imshow(im_bright);

%% Histogram Equalization

% figure; imhist(im);
% im_histeq = histeq(im);

% figure;
% subplot(121); imshow(im);
% subplot(122); imshow(im_histeq);
% 
% figure; imhist(im_histeq);

%% pixel windowing

% R = im(:, :, 1);
% B = im(:, :, 2);
% G = im(:, :, 3);
% 
% R(100 < R & R > 120) = 0;
% B(100 < B & B > 120) = 0;
% G(100 < G & G > 120) = 0;
% 
% I(:, :, 1) = R;
% I(:, :, 2) = B;
% I(:, :, 3) = G;
% 
% imshow(I);
% 
% im_gray = rgb2gray(im);
% im_gray(100 < im_gray & im_gray > 120) = 0;
% 
% imshow(im_gray);

%% Inversion

% im_inv = 255 - im; 
% figure; 
% subplot(121); imshow(im);
% subplot(122); imshow(im_inv);

%% Gamma Correction

% im_gamma = mat2gray(((double(im)/255).^(1.5)).*255);
% figure;
% subplot(121); imshow(im);
% subplot(122); imshow(im_gamma);

%% Im Derivatives and Gradients (1st and 2nd)

% im = imread('apples.JPG');
% 
% K = [-1 1]';
% im_gray = rgb2gray(im);
% im_x = conv2(im_gray, K);
% figure;
% subplot(121); imshow(im);
% subplot(122); imshow(im_x);
% 
% im_g = imgradient(im_gray);
% figure;imshow(im_g); 

%% Edge Detection (Canny)



%% gaussian blur



%% DoG


%% LoG



%% Keypoint


%% Blob Detection










