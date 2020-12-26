% IET MATLAB WORKSHOP 2020
% Author : Ashwin De Silva

%% Signal Processing Basic Concepts

clear all;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Simple Sinusoidal Wave

% Fs = 1000; % sampling frequency in Hz
% 
% % time vector
% t = 0:1/Fs:1;
% 
% A = 1;
% omega_0 = 2*pi*2;
% phi = pi/2;
% 
% % signal
% x = A*cos(omega_0*t + phi);
% 
% % plotting
% plot(t, x);
% xlabel('Time');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% High Frequency Noise

% Fs = 200; % sampling frequency in Hz
% 
% % time vector
% t = 0:1/Fs:1;
% 
% % signal
% y = sin(2*pi*5*t) + sin(2*pi*50*t) ;
% figure; plot(t, y);
% xlabel('Time');
% 
% figure; spectrogram(y, 128, 120, 128, 200, 'yaxis');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Random Noise

% Fs = 1000; % sampling frequency in Hz
% 
% % time vector
% t = 0:1/Fs:1;
% 
% % signal
% y = sin(2*pi*5*t);
% 
% % adding addtive gaussian noise
% y_n = y + 0.5*randn(size(t));
% 
% plot(t, y_n);
% xlabel('Time');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Exponential Functions

% Fs = 100; % sampling frequency in Hz
% 
% % time vector
% t = -10:1/Fs:10;
% 
% C = 0.1;
% a = 0.3;
% 
% % signal
% y = C*exp(a*t);
% 
% % plotting
% plot(t, y);
% xlabel('Time');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Impulse, Step, Ramp and Quad Functions

% Fs = 100; % sampling frequency in Hz
% 
% % time vector
% t = -1:1/Fs:1;
% 
% % signals
% impulse = t==0;
% unitstep = t >= 0;
% ramp = t.*unitstep;
% quad = t.^2.*unitstep;
% 
% % plotting
% plot(t, impulse);hold on;
% plot(t, unitstep);hold on;
% plot(t, ramp);hold on;
% plot(t, quad);hold on;
% xlabel('Time');
% legend(['impulse', 'step', 'ramp', 'quad']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Square Functions

% Fs = 100; % sampling frequency in Hz
% 
% % time vector
% t = -1:1/Fs:1;
%  
% A = 2;
% 
% % signal
% y = A*square(2*pi*2*t);
% 
% % plotting
% plot(t, y);
% xlabel('Time');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Sawtooth Functions

% Fs = 100; % sampling frequency in Hz
% 
% % time vector
% t = 0:1/Fs:2;
%  
% A = 1.5;
% 
% % signal
% y = A*sawtooth(2*pi*2*t);
% 
% % plotting
% plot(t, y);
% xlabel('Time');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Chirp Signal

% Fs = 1000; % sampling frequency in Hz
% 
% % time vector
% t = 0:1/Fs:2;
% 
% % signal
% y = chirp(t, 0, 1, 150); 
% % starting frequency : 0 Hz, at t = 1s : 150 Hz
% 
% % plotting
% figure; plot(t, y);
% xlabel('Time');
% 
% % spectogram
% figure; spectrogram(y, 256, 250, 256, 1000, 'yaxis');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Sinc Fucntion

% Fs = 1000; % sampling frequency in Hz
% 
% % time vector
% t = -5:1/Fs:5;
% 
% % signal
% y = sinc(t);
% 
% % plotting
% figure; plot(t, y);
% xlabel('Time');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Triangular Pulse

% Fs = 1000; % sampling frequency in Hz
% 
% % time vector
% t = -1:1/Fs:1;
% 
% % signal
% y = tripuls(t, 0.5);
% 
% % plotting
% figure; plot(t, y);
% xlabel('Time');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Discrete-Time Sinusoid

% N = 12;
% Omega_0 = 2*pi/N;
% n = -1*N:1:N*1;
% phi = 0; 
% A = 1;
% xn = A*cos(Omega_0*n + phi);
% 
% stem(n, xn);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Discrete-Time Real Exponentials

% nmax = 10;
% n = -nmax:nmax;
% C=1;
% alpha = -0.9;
% xn = C*alpha.^n;
% 
% stem(n, xn);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Discrete-Time Complex Exponentials

% nmax = 20;
% n = -nmax:nmax;
% 
% theta = 0;
% C=1*exp(j*theta);
% 
% Omega_0 = 2*pi/16;
% 
% alpha = 0.9*exp(j*Omega_0);
% 
% xn = C.*alpha.^n;
% 
% stem(n, real(xn));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Fourier Transform (N-Point)

% Fs = 1000;      % sampling frequency
% L = 1500;       % length of the signal
% T = 1/Fs;       % sampling period
% 
% t = (0:L-1)*T;  % time vector
% 
% y = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);   % signal
% 
% % plotting
% figure; plot(t, y);
% xlabel('Time(s)');
% 
% % computing the fourier transform of the signal
% 
% L = length(y); % length / number of samples of the signal
% NFFT = 2^nextpow2(L); % selecting the next power of 2 after L (typically)
% Y = fft(y, NFFT); % getting the FFT
% P2 = abs(Y/L); % we take the magitude and scale to get the double sided spectrum
% 
% % we only take half of the values since for a real valued signal the DFT
% % has the conjugate symmetry. We multiply the values by two to account for
% % the power we gave up. (but not the fft(1) since it corresponds to the DC
% % values of the signal and is common to both spectrums.
% 
% P1 = P2(1:NFFT/2+1); 
% P1 = 2*P1;
% P1(1) = P1(1)/2; % now P1 is the single sided spectrum
% 
% % P(1) corresponds to the DC value, P(end) correspond to the value at 
% % nyquist frequency
% 
% % frequencies 
% f = Fs/2*linspace(0, 1, NFFT/2+1);
% 
% % plotting 
% figure; plot(f, P1);
% xlabel('Frequency (Hz)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Fourier Transform - Double Sided Spectrum

% Fs = 500;      % sampling frequency
% L = 1500;       % length of the signal
% T = 1/Fs;       % sampling period
% 
% t = (0:L-1)*T;  % time vector
% 
% y = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);   % signal
% 
% % plotting
% figure; plot(t, y);
% xlabel('Time(s)');
% 
% % computing the fourier transform of the signal
% 
% L = length(y); % length / number of samples of the signal
% Y = fft(y); % getting the FFT
% Y = fftshift(Y); % shifts the DC value to the middle of the array
% 
% M = abs(Y/L); % we take the magitude and scale to get the double sided spectrum
% Y(abs(Y) < 1e-6) = 0; % remove the small values
% P = unwrap(angle(Y));
% 
% % frequencies 
% f = Fs/2*linspace(-1, 1, L);
% 
% % plotting 
% figure; 
% subplot(211); plot(f, M);
% ylabel('|Y(f)|');
% xlabel('Frequency (Hz)');
% subplot(212); plot(f, P);
% ylabel('Phase(radian)');
% xlabel('Frequency (Hz)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Fourier Transform

% Fs = 1000;      % sampling frequency
% L = 1500;       % length of the signal
% T = 1/Fs;       % sampling period
% 
% t = (0:L-1)*T;  % time vector
% 
% y = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);   % signal
% 
% plotting
% figure; plot(t, y);
% xlabel('Time(s)');
% 
% computing the fourier transform of the signal
% 
% L = length(y); % length / number of samples of the signal
% NFFT = L;
% Y = fft(y, NFFT); % getting the FFT
% P2 = abs(Y/L); % we take the magitude and scale to get the double sided spectrum
% 
% we only take half of the values since for a real valued signal the DFT
% has the conjugate symmetry. We multiply the values by two to account for
% the power we gave up. (but not the fft(1) since it corresponds to the DC
% values of the signal and is common to both spectrums.
% 
% P1 = P2(1:L/2+1); 
% P1 = 2*P1;
% P1(1) = P1(1)/2; % now P1 is the single sided spectrum
% 
% P(1) corresponds to the DC value, P(end) correspond to the value at 
% nyquist frequency
% 
% frequencies 
% f = Fs/2*linspace(0, 1, L/2+1);
% 
% plotting 
% figure; plot(f, P1);
% xlabel('Frequency (Hz)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Fourier Transform - Phase

% Fs = 1000;      % sampling frequency
% L = 1500;       % length of the signal
% T = 1/Fs;       % sampling period
% 
% t = (0:L-1)*T;  % time vector
% 
% y = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);   % signal
% 
% % plotting
% figure; plot(t, y);
% xlabel('Time(s)');
% 
% % computing the fourier transform of the signal
% 
% L = length(y); % length / number of samples of the signal
% NFFT = L;
% Y = fft(y, NFFT); % getting the FFT
% Y(abs(Y)<1e-6) = 0;
% P2 = unwrap(angle(Y)); % we take the unwrapped phase
% 
% P1 = P2(1:L/2+1); 
% P1 = 2*P1;
% P1(1) = P1(1)/2; % now P1 is the single sided spectrum
% 
% % P(1) corresponds to the DC value, P(end) correspond to the value at 
% % nyquist frequency
% 
% % frequencies 
% f = Fs/2*linspace(0, 1, L/2+1);
% 
% % plotting 
% figure; plot(f, P1);
% xlabel('Frequency (Hz)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Fourier Transform - Gaussian Pulse

% Fs = 1000;
% t = -0.5:1/Fs:0.5;
% 
% % signal
% y = 1/sqrt(2*pi*0.1^2)*exp(-t.^2/(2*0.1^2));
% 
% % plotting
% figure; plot(t, y);
% xlabel('Time(s)');
% 
% % Fourier Transform
% Y = fft(y);
% L = length(y);
% P2 = abs(Y/L);
% P1 = 2*P2(1:L/2+1);
% P1(1) = P1(1)/2;
% 
% f = Fs/2*linspace(0,1,L/2+1);
% 
% figure; plot(f, P1);
% xlabel('Frequency (Hz)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Fourier Transform - Cosine Signals

% Fs = 100;
% t = 0:1/Fs:1;
% 
% % signal
% x1 = 1*cos(2*pi*10*t);
% x2 = 2*cos(2*pi*20*t);
% x3 = 1.5*cos(2*pi*30*t);
% 
% X = [x1', x2', x3'];
% 
% % plotting
% figure;
% for i = 1:3
%     subplot(3, 1, i); plot(t, X(:, i));
% end
% xlabel('Time(s)');
% 
% % Fourier Transform
% Y = fft(X);
% L = size(X, 1);
% P2 = abs(Y/L);
% P1 = 2*P2(1:L/2+1, :);
% P1(1, :) = P1(1, :)/2;
% 
% f = Fs/2*linspace(0,1,L/2+1);
% 
% % plotting
% figure;
% for i = 1:3
%     subplot(3, 1, i); plot(f, P1(:, i));
% end
% xlabel('Time(s)');
% xlabel('Frequency (Hz)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Fourier Transform - IFFT

% Fs = 100;
% t = 0:1/Fs:1;
% 
% % signal
% x = 2*cos(2*pi*5*t);
% x_re = real(ifft(fft(x)));
% 
% figure;
% subplot(121); plot(t, x);
% title('Original');
% subplot(122); plot(t, x_re); 
% ylim([-2, 2]);
% title('Reconstructed');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Spectrum Analysis using FFT

% rng default
% 
% Fs = 1000;      % sampling frequency
% L = 1000;       % length of the signal
% T = 1/Fs;       % sampling period
% 
% t = (0:L-1)*T;  % time vector
% 
% y = cos(2*pi*100*t) + randn(size(t));  % noisy signal
% 
% figure; plot(t, y);
% xlabel('Time(s)');
% 
% L = length(y); % length / number of samples of the signal
% Y = fft(y); % getting the FFT
% f = Fs*linspace(0, 1, L/2+1);
% 
% psdy = (1/(Fs*L)) * abs(Y(1:L/2+1)).^2;
% psdy(2:end-1) = 2*psdy(2:end-1);
% 
% figure; plot(f, 10*log10(psdy));
% xlabel('frequency(Hz)');
% ylabel('power/freq (dB/Hz)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Spectrum Analysis using FFT

% rng default
% 
% Fs = 1000;      % sampling frequency
% L = 1000;       % length of the signal
% T = 1/Fs;       % sampling period
% t = (0:L-1)*T;  % time vector
% 
% y = cos(2*pi*100*t) + randn(size(t));  % noisy signal
% 
% figure; plot(t, y);
% xlabel('Time(s)');
% 
% f = Fs/2*linspace(0, 1, L/2+1);
% psdy = periodogram(y,[],f,Fs); 
% 
% figure; plot(f, pow2db(psdy));
% xlabel('frequency(Hz)');
% ylabel('power/freq (dB/Hz)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Meauring the power of a signal

% rng default
% 
% Fs = 1000;      % sampling frequency
% L = 1000;       % length of the signal
% T = 1/Fs;       % sampling period
% t = (0:L-1)*T;  % time vector
% 
% sigma = 0.01; % noise variance
% y = chirp(t,100,1,300)+sigma*randn(size(t));
% 
% band power using RMS
% power_rms = rms(y)^2;
% disp('using RMS : ');
% disp(power_rms);
% 
% band power using bandpower function
% power_func = bandpower(y, Fs, [0, Fs/2]);
% disp('using the matlab func. : ');
% disp(power_func);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Meauring the power of a signal in difference frequency bands

% load(fullfile(matlabroot,'examples','signal','AmpOutput.mat'));
% Fs = 3600;
% y = y-mean(y);
% L = length(y);
% 
% t = (0:L-1)*1/Fs;
% figure; subplot(211); plot(t, y);
% xlabel('Time(s)');
% 
% Y = fft(y);
% P2 = abs(Y/L);
% P1 = 2*P2(1:L/2+1);
% P1(1) = P1(1)/2;
% 
% f =  Fs/2*linspace(0, 1, L/2+1);
% subplot(212); plot(f, P1);
% xlabel('Frequency(Hz)');
% 
% %%%%% start code %%%%
% 
% freq = [60, 120, 180]; % harmonics of 60 Hz powerline noise
% band_power = [];
% 
% for i = 1:3
%     band_power(i) = bandpower(y, Fs, [freq(i)-5, freq(i)+5]);
% end
% 
% total_power = bandpower(y, Fs, [0, Fs/2]);
% 
% power_percentage = band_power/total_power * 100;
% 
% power_db = pow2db(band_power);
% 
% %%%% end code %%%%
% 
% Names = {'60'; '120'; '180'}; 
% T = table(freq', band_power' , power_percentage', power_db', 'RowNames', Names)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SNR

% Fs = 1000;
% t = (0:1/Fs:2);
% 
% y = 2*cos(2*pi*10*t);
% % n = 0.1*randn(size(t));
% n = 0.001*cos(2*pi*1000*t);
% y_n = y + n;
% 
% %%%% start code %%%%
% 
% SNR_manual = 10*log10(rms(y)^2/rms(n)^2);
% 
% SNR = snr(y, n);
% 
% %%%% start code %%%%
% 
% disp('Manual SNR : ');
% disp(SNR_manual);
% 
% disp('SNR : ');
% disp(SNR);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Convolution

% tu = (-10:1:10);
% u = (tu<4) & (tu>-4);
% 
% tv = (-5:1:5);
% v = 5-abs(tv);
% 
% %%%% start code %%%%
% 
% % x = conv(u, v);
% X = fft(u, 31).*fft(v, 31);
% x_f = ifft(X);
% 
% %%%% end code %%%%
% 
% tx = (-15:1:15);
% plot(tx, x_f);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 3rd order Moving Average Filter with Convolution

% Fs = 50;
% t = -5:1/Fs:5;
% 
% x = 1/sqrt(2*pi*0.3^2)*exp(-t.^2/(2*0.3^2)) + 0.01*randn(size(t));
% subplot(121); plot(t, x); title('Input');
% 
% %%%% start code %%%%
% 
% h = [1 1 1 1]/4; % 3rd order moving avg. filter impulse response
% y = conv(x, h, 'same');
% 
% %%%% end code %%%%
% 
% subplot(122);plot(t, y); title('Output');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 3rd order Moving Average Filter with Filter Function

% Fs = 50;
% t = -5:1/Fs:5;
% 
% x = 1/sqrt(2*pi*0.3^2)*exp(-t.^2/(2*0.3^2)) + 0.01*randn(size(t));
% subplot(121); plot(t, x); title('Input');
% 
% %%%% start code %%%%
% 
% b = [0.25 0.25 0.25 0.25];
% a = 1;
% y = filter(b, a, x);
% 
% %%%% end code %%%%
% 
% subplot(122);plot(t, y); title('Output');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 2nd order Autoregressive Filter with Filter Function

% Fs = 50;
% t = -5:1/Fs:5;
% 
% x = 1/sqrt(2*pi*0.3^2)*exp(-t.^2/(2*0.3^2)) + 0.01*randn(size(t));
% subplot(121); plot(t, x); title('Input');
% 
% %%%% start code %%%%
% 
% b = 1;
% a = [1 -0.9];
% fvtool(b,a);
% y = filter(b, a, x);
% 
% %%%% end code %%%%
% 
% subplot(122);plot(t, y); title('Output');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FIR filter without no window

% %%%% start code %%%%
% 
% n = -25:1:25;
% fc = 0.4;
% b = fc*sinc(fc*n);
% a = 1;
% 
% fvtool(b, a);
% 
% %%%% end code %%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FIR filter with Hamming window

%%%% start code %%%%

% n = -25:1:25;
% fc = 0.4;
% b = fc*sinc(fc*n);
% b = b.*hamming(51)'; % applying the hamming window
% a = 1;
% 
% fvtool(b, a);

%%%% end code %%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FIR filter with Hamming window

% Fs = 1000;
% t = 0:1/Fs:1;
% 
% x = cos(2*pi*50*t) + cos(2*pi*100*t) + cos(2*pi*200*t); 
% 
% %%%% start code %%%%
% f_l = 80/(Fs/2);
% f_u = 180/(Fs/2);
% 
% % b = fir1(48, [f_l]); % low-pass
% % b = fir1(48, [f_l, f_u]); % band-pass
% b = fir1(48, [f_u], 'high'); % high-pass
% 
% % fvtool(b, 1);
% 
% y = filter(b, 1, x);
% 
% %%%% end code %%%%
% 
% Y = fft(y);
% L = length(y);
% P2 = abs(Y/L);
% P1 = 2*P2(1:L/2+1);
% P1(1) = P1(1)/2;
% 
% f = Fs/2*linspace(0, 1, L/2+1);
% 
% subplot(211); plot(t, x);
% subplot(212); plot(f, P1); xlabel('Frequency(Hz)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FIR filter with Kaiser window

%%%% start code %%%%

% fsamp = 8000; % sampling freq
% fcuts = [1000 1500]; % frequency band edgess
% mags = [1 0]; % magnitudes of the pass and stop bands
% devs = [0.05 0.01]; % pass-band ripple and stop band attenuation
% 
% [n,Wn,beta,ftype] = kaiserord(fcuts,mags,devs,fsamp);
% hh = fir1(n,Wn,ftype,kaiser(n+1,beta),'noscale');
% 
% fvtool(hh, 1)

%%%% end code %%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Classical IIR Filters

%%%% start code %%%%

% [b, a] = butter(5, 0.5);
% fvtool(b, a); hold on;
% [b, a] = cheby1(5, 1, 0.5);
% fvtool(b, a); hold on;
% [b, a] = cheby2(5, 60, 0.5);
% fvtool(b, a); hold on;
% [b, a] = ellip(5, 1, 60, 0.5);
% fvtool(b, a); hold off;

%%%% end code %%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Butterworth Filter Example

% Fs = 500;
% t = (0:1/Fs:1);
% 
% x = sin(2*pi*10*t) + 0.2*sin(2*pi*180*t);
% 
% %%%% start code %%%%
% 
% [n,Wn] = buttord(100/(Fs/2), 120/(Fs/2), 3, 60); 
% [z, p, k] = butter(n, Wn);
% sos = zp2sos(z, p, k);
% fvtool(sos);
% y = sosfilt(sos, x);
% 
% %%%% end code %%%%
% 
% subplot(211); plot(t, x); title('Noisy');
% subplot(212); plot(t, y); title('Filteres');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Kaiser Window Example using designfilt

% Fs = 500;
% t = (0:1/Fs:1);
% 
% x = sin(2*pi*10*t) + 0.2*sin(2*pi*180*t);
% 
% %%%% start code %%%%
% 
% lpFilt = designfilt('lowpassfir', ...
%          'PassbandFrequency', 100, 'StopbandFrequency', 120, ...
%          'PassbandRipple', 3, 'StopbandAttenuation', 60, 'SampleRate', Fs, ...
%          'DesignMethod', 'kaiserwin');
% fvtool(lpFilt)
% 
% y = filter(lpFilt, x);
% 
% %%%% end code %%%%
% 
% subplot(211); plot(t, x); title('Noisy');
% subplot(212); plot(t, y); title('Filteres');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Butterworth Filter Example using designfilt

% Fs = 500;
% t = (0:1/Fs:1);
% 
% x = sin(2*pi*10*t) + 0.2*sin(2*pi*180*t);
% 
% %%%% start code %%%%
% 
% lpFilt = designfilt('lowpassiir', ...
%          'PassbandFrequency', 100, 'StopbandFrequency', 120, ...
%          'PassbandRipple', 3, 'StopbandAttenuation', 60, 'SampleRate', Fs, ...
%          'DesignMethod', 'butter');
% fvtool(lpFilt)
% 
% y = filter(lpFilt, x);
% 
% %%%% end code %%%%
% 
% subplot(211); plot(t, x); title('Noisy');
% subplot(212); plot(t, y); title('Filteres');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Sampling and Reconstrcution

% % analog signal
% Fs = 50;
% t = -5:1/Fs:5;
% x = funx(t);
% [X, f] = double_sided_fourier(x, Fs);
% 
% figure; plot(t, x);
% figure; plot(f, X);
% 
% % sampling
% fs = 12;
% Ts = 1/fs;
% t_n = -5:Ts:5;
% n = -(length(t_n)-1)/2:(length(t_n)-1)/2;
% x_n = funx(t_n);
% figure; stem(n, x_n);
% 
% [X, f] = double_sided_fourier(x_n, fs);
% figure; plot(f, X);
% 
% % reconstruction
% x_r = zeros(size(t));
% for n = -1000:1000
%     x_r = x_r + funx(n*Ts)*sinc((t-n*Ts)/Ts);
% end
% 
% figure; plot(t, x_r);
% 
% figure; 
% plot(t, x, 'LineWidth', 1); hold on;
% stem(t_n, x_n, 'r'); hold off
% legend('x(t)', 'x[n]');
% 
% figure; 
% plot(t, x, 'LineWidth', 1); hold on;
% plot(t, x_r, '--r'); hold off
% legend('x(t)', 'x_r(t)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Effect of Aliasing

% load handel.mat
% hfile = 'handel.wav';
% audiowrite(hfile, y, Fs); % save the audio file
% 
% [x, fs] = audioread(hfile);
% sound(x, fs); % play the original sound
% pause; % Press “Enter”
% 
% [x, fs] = audioread(hfile);
% y= downsample(x, 4);
% sound(y,fs/4); % Down sampled signal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Inverse Z-Transform

% num = [1 4 2];
% den = [1 -0.25];
% 
% uimp = @(x) (x==0); % unit impulse
% 
% n = -50:50;
% x = filter(num, den, uimp(n));
% 
% stem(n, x);
% 
% fvtool(num, den);

%% Discrete-Time Filtering

% num = [4.43e-4 8.86e-4 4.43e-4];
% den = [1 -1.94 0.94];

num = [1];
den = [1 -0.9];

uimp = @(x) (x==0); % unit impulse
ustep = @(x) (x>=0); % unit step

ni = -50:50;
hn = filter(num, den, uimp(ni)); % impulse response
figure; stem(ni, hn); title('Impulse Response');

n = [0:200];
xn = ustep(n+100) - ustep(n-100);
figure; stem(n, xn); title('x[n]');

% % filter 
% yn = zeros(1,201);  % Initialize
% for i = 3:length(yn)
%     yn(i) = 4.43e-4*xn(i) + 8.86e-4*xn(i-1) + 4.43e-4*xn(i-2) ... 
%             + 1.94*yn(i-1) - 0.94*yn(i-2);
% end

% filter
yn = zeros(1,201);  % Initialize
for i = 3:length(yn)
    yn(i) = xn(i) + 0.9*yn(i-1);
end

figure; stem(n, yn); title('y[n]');

% filter 
ynn = filter(num, den, xn);
figure; stem(n, ynn); title('y[n]');

fvtool(num, den);

% support functions ----

function x = funx(t)
    x = (1.5 + 0.3*sin(2*pi*t) + sin(2*pi*1/3*t) - sin(2*pi*1/10*t)).*sinc(5*t);
%     x = (1.5 + 0.3*sin(2*pi*t) + sin(2*pi*1/3*t) - sin(2*pi*1/10*t)).*gauspuls(t, 5, 0.7);
end

function [X, f] = single_sided_fourier(x, Fs)
    K = fft(x);
    L = length(x);
    P2 = abs(K/L);
    P1 = 2*P2(1:L/2+1);
    P1(1) = P1(1)/2;
    X = P1;
    
    f = Fs/2*linspace(-1,1,L/2+1);
end

function [X, f] = double_sided_fourier(x, Fs)
    K = fft(x);
    K = fftshift(K);
    L = length(x);
    P2 = abs(K/L);
    X = P2;
    f = Fs/2*linspace(-1,1,L);
end


