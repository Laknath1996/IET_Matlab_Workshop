% IET MATLAB WORKSHOP 2020
% Author : Ashwin De Silva

%% Signal Processing Basic Concepts

clear all;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Analog Modulation and Demodulation

% Am = 1;                 % Remember the constraint [Am <= 1]
% u = 0.6;                % modulation index
% Ac = 1;                 % carrier amplitude
% Cf = 2E6;               % Carrier frequency;
% Sf = 50E3;              % Signal frequency;
% 
% Ts = 1E-8;              % sampling frequency
% fs = 1/Ts;
% t = 0:Ts:0.0001;
% Wc = 2*pi*Cf;           % carrier freq in rad/s
% Ws = 2*pi*Sf;           % signal freq in rad/s
% 
% %%%% start code %%%%
% 
% % generate the signals
% ct = Ac*cos(Wc*t);      % carrier signal
% mt = Am*cos(Ws*t);      % message signal
% mt = mt / max(abs(mt)); % normalized message signal
% 
% % modulation
% xmt = (1 + u*mt).*ct;    % modulated signal
% 
% %%%% start code %%%%
% 
% % time domain plots 
% figure;
% subplot(311); plot(t, ct); title('Carrier Signal');
% axis([0 0.0001 -2 2]);
% subplot(312); plot(t, mt); title('Message Signal');
% axis([0 0.0001 -2 2]);
% subplot(313); plot(t, xmt); title('Modulated Signal');
% axis([0 0.0001 -2 2]);
% 
% % frequency domain
% [Xc, f] = double_sided_fourier(ct, fs);
% [Xm, ~] = double_sided_fourier(mt, fs);
% [Xmod, ~] = double_sided_fourier(xmt, fs);
% 
% % frequency domain plots 
% figure;
% subplot(311); plot(f, Xc); title('Carrier Signal');
% axis([-2.5E6 2.5E6 0 1]);
% subplot(312); plot(f, Xm); title('Message Signal');
% axis([-2.5E6 2.5E6 0 1]);
% subplot(313); plot(f, Xmod); title('Modulated Signal');
% axis([-2.5E6 2.5E6 0 1]);
% 
% % demodulation
% z = xmt.*cos(2*pi*Cf*t);
% [b, a] = butter(5, Cf/(fs/2));
% z = filtfilt(b, a, z)*2;
% xdmt = (z - Ac)/u;
% 
% % % demodulation
% % z = xmt;
% % z(z < 0) = 0;
% % [b, a] = butter(5, 2*Sf/(fs/2));
% % xdmt = filtfilt(b, a, z)*2;
% % xdmt = (xdmt - Ac)/u;
% 
% % time domain plots with demodulation
% figure;
% subplot(411); plot(t, ct); title('Carrier Signal');
% axis([0 0.0001 -2 2]);
% subplot(412); plot(t, mt); title('Message Signal');
% axis([0 0.0001 -2 2]);
% subplot(413); plot(t, xmt); title('Modulated Signal');
% axis([0 0.0001 -2 2]);
% subplot(414); plot(t, xdmt); title('Demodulated Signal');
% axis([0 0.0001 -2 2]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Frequency Modulation and Demodulation

% Am = 1;                 % message amplitude
% beta = 20;              % modulation index
% Ac = 1;                 % carrier amplitude
% Cf = 3E3;               % Carrier frequency;
% Sf = 1E2;               % Signal frequency;
% 
% Ts = 2E-5;              % sampling frequency
% fs = 1/Ts;
% t = 0:Ts:1;
% Wc = 2*pi*Cf;           % carrier freq in rad/s
% Ws = 2*pi*Sf;           % signal freq in rad/s
% 
% %%%% start code %%%%
% ct = Ac*cos(Wc*t);      % carrier signal
% mt = Am*cos(Ws*t);      % message signal
% 
% % generate the signals
% phi = Wc*t + beta*sin(Ws*t); % modulated phase
% 
% % modulation
% xmt = Ac*cos(phi);    % modulated signal
% 
% %%% end code %%%
% 
% % time domain plots 
% figure;
% subplot(311); plot(t, ct); title('Carrier Signal');
% axis([0 0.05 -2 2]);
% subplot(312); plot(t, mt); title('Message Signal');
% axis([0 0.05 -2 2]);
% subplot(313); plot(t, xmt); title('Modulated Signal');
% axis([0 0.05 -2 2]);
% 
% % frequency domain
% [Xc, f] = double_sided_fourier(ct, fs);
% [Xm, ~] = double_sided_fourier(mt, fs);
% [Xmod, ~] = double_sided_fourier(xmt, fs);
% 
% % frequency domain plots 
% figure;
% subplot(311); plot(f, Xc); title('Carrier Signal');
% axis([-0.1E5 0.1E5 0 0.8]);
% subplot(312); plot(f, Xm); title('Message Signal');
% axis([-0.1E5 0.1E5 0 0.8]);
% subplot(313); plot(f, Xmod); title('Modulated Signal');
% axis([-0.1E5 0.1E5 0 0.8]);
% 
% % % demodulation
% % xdmt = fmdemod(xmt, Cf, Sf, beta*Sf/Am);
% 
% % % time domain plots with demodulation
% % figure;
% % subplot(411); plot(t, ct); title('Carrier Signal');
% % axis([0 0.05 -2 2]);
% % subplot(412); plot(t, mt); title('Message Signal');
% % axis([0 0.05 -2 2]);
% % subplot(413); plot(t, xmt); title('Modulated Signal');
% % axis([0 0.05 -2 2]);
% % subplot(414); plot(t, xdmt); title('Demodulated Signal');
% % axis([0 0.05 -2 2]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Amplitude Shift Keying 

% bit_pattern = [ 1 0 0 1 1 0 1 1 1 0 1 1 0 0 0 1 1 1 1 0 0 0 1 1 ];
% ask(bit_pattern, 2); % bi-level
% ask(bit_pattern, 4); % 4-level`

%% Frequency Shift Keying 

% bit_pattern = [ 1 0 0 1 1 0 1 1 1 0 1 1 0 0 0 1 1 1 1 0 0 0 1 1 ];
% fsk(bit_pattern, 2, [0.8E6, 2.4E6]); % bi-level
% fsk(bit_pattern, 4, [0.8E6, 2.4E6, 4.8E6, 7.2E6]); % 4-level

%% Phase Shift Keying 

% bit_pattern = [ 1 0 0 1 1 0 1 1 1 0 1 1 0 0 0 1 1 1 1 0 0 0 1 1 ];
% psk(bit_pattern, 2); % bi-level
% psk(bit_pattern, 4); % 4-level

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% support functions ----

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

%% Amplitude Shift-Keying (ASK)

function [] = ask(bit_pattern,n)
    Cf = 1.2E6;                     % Carrier frequency 1.2 MHz;

    delt = 1E-8;
    fs = 1/delt;
    
    samples_per_bit=250;
    tmax = (samples_per_bit*length(bit_pattern)-1)*delt;
    t= 0:delt:tmax; % Time window we are interested in
    
    % Generation of the binary info signal
    bits=zeros(1,length(t));
    for bit_no=1:1:length(bit_pattern)
        for sample=1:1:samples_per_bit 
            bits((bit_no-1)*samples_per_bit+sample)=bit_pattern(bit_no);
        end
    end
    
    % See what it looks like
    figure;
    subplot(2,1,1);plot(t,bits);
    ylabel('Amplitude');
    title('Info signal');
    axis([0 tmax -2 2]);
    grid on;
    
    % ASK modulation
    ASK=[];
    if n==2
        for bit_no=1:1:length(bit_pattern)
            if bit_pattern(bit_no)==1
                t_bit = (bit_no- 1)*samples_per_bit*delt:delt:(bit_no*samples_per_bit-1)*delt;
                Wc = 2*pi*Cf*t_bit;
                mod = (1)*sin(Wc);
            elseif bit_pattern(bit_no)==0
                t_bit = (bit_no- 1)*samples_per_bit*delt:delt:(bit_no*samples_per_bit-1)*delt;
                Wc = 2*pi*Cf*t_bit;
                mod = (0)*sin(Wc);
            end
            ASK=[ASK mod];
        end
    elseif n==4
        for bit_no=2:2:length(bit_pattern)
            disp(bit_pattern(bit_no-1: bit_no));
            if bit_pattern(bit_no-1: bit_no)==[0, 0]
                t_bit = (bit_no-2)*samples_per_bit*delt:delt:((bit_no)*samples_per_bit-1)*delt;
                Wc = 2*pi*Cf*t_bit;
                mod = (0)*sin(Wc);
            elseif bit_pattern(bit_no-1: bit_no)==[0, 1]
                t_bit = (bit_no-2)*samples_per_bit*delt:delt:((bit_no)*samples_per_bit-1)*delt;
                Wc = 2*pi*Cf*t_bit;
                mod = (1/3)*sin(Wc);
            elseif bit_pattern(bit_no-1: bit_no)==[1, 1]
                t_bit = (bit_no-2)*samples_per_bit*delt:delt:((bit_no)*samples_per_bit-1)*delt;
                Wc = 2*pi*Cf*t_bit;
                mod = (2/3)*sin(Wc);
            else
                t_bit = (bit_no-2)*samples_per_bit*delt:delt:((bit_no)*samples_per_bit-1)*delt;
                Wc = 2*pi*Cf*t_bit;
                mod = (1)*sin(Wc);
            end
            ASK=[ASK mod];
        end
    end
    subplot(2,1,2); plot(t,ASK);
    ylabel('Amplitude');
    title('ASK Modulated Signal');
    axis([0 tmax -2 2]);
    grid on;
end


%% Frequency Shift Keying

function [] = fsk(bit_pattern, n, Fe)               

    delt = 1E-8;
    fs = 1/delt;
    
    samples_per_bit=250;
    tmax = (samples_per_bit*length(bit_pattern)-1)*delt;
    t= 0:delt:tmax; % Time window we are interested in
    
    % Generation of the binary info signal
    bits=zeros(1,length(t));
    for bit_no=1:1:length(bit_pattern)
        for sample=1:1:samples_per_bit 
            bits((bit_no-1)*samples_per_bit+sample)=bit_pattern(bit_no);
        end
    end
    
    % See what it looks like
    figure;
    subplot(2,1,1);plot(t,bits);
    ylabel('Amplitude');
    title('Info signal');
    axis([0 tmax -2 2]);
    grid on;
    
    % FSK modulation
    FSK=[];
    if n==2
        for bit_no=1:1:length(bit_pattern)
            if bit_pattern(bit_no)==0
                t_bit = (bit_no- 1)*samples_per_bit*delt:delt:(bit_no*samples_per_bit-1)*delt;
                Wc = 2*pi*Fe(1)*t_bit;
                mod = (1)*sin(Wc);
            elseif bit_pattern(bit_no)==1
                t_bit = (bit_no- 1)*samples_per_bit*delt:delt:(bit_no*samples_per_bit-1)*delt;
                Wc = 2*pi*Fe(2)*t_bit;
                mod = (1)*sin(Wc);
            end
            FSK=[FSK mod];
        end
    elseif n==4
        for bit_no=2:2:length(bit_pattern)
            disp(bit_pattern(bit_no-1: bit_no));
            if bit_pattern(bit_no-1: bit_no)==[0, 0]
                t_bit = (bit_no-2)*samples_per_bit*delt:delt:((bit_no)*samples_per_bit-1)*delt;
                Wc = 2*pi*Fe(1)*t_bit;
                mod = (1)*sin(Wc);
            elseif bit_pattern(bit_no-1: bit_no)==[0, 1]
                t_bit = (bit_no-2)*samples_per_bit*delt:delt:((bit_no)*samples_per_bit-1)*delt;
                Wc = 2*pi*Fe(2)*t_bit;
                mod = (1)*sin(Wc);
            elseif bit_pattern(bit_no-1: bit_no)==[1, 1]
                t_bit = (bit_no-2)*samples_per_bit*delt:delt:((bit_no)*samples_per_bit-1)*delt;
                Wc = 2*pi*Fe(3)*t_bit;
                mod = (1)*sin(Wc);
            else
                t_bit = (bit_no-2)*samples_per_bit*delt:delt:((bit_no)*samples_per_bit-1)*delt;
                Wc = 2*pi*Fe(4)*t_bit;
                mod = (1)*sin(Wc);
            end
            FSK=[FSK mod];
        end
    end
    subplot(2,1,2); plot(t,FSK);
    ylabel('Amplitude');
    title('FSK Modulated Signal');
    axis([0 tmax -2 2]);
    grid on;
end


%% Phase Shift Keying

function [] = psk(bit_pattern, n)
    Cf = 4E5;                     % Carrier frequency 1.2 MHz;

    delt = 1E-8;
    fs = 1/delt;
    
    samples_per_bit=250;
    tmax = (samples_per_bit*length(bit_pattern)-1)*delt;
    t= 0:delt:tmax; % Time window we are interested in
    
    % Generation of the binary info signal
    bits=zeros(1,length(t));
    for bit_no=1:1:length(bit_pattern)
        for sample=1:1:samples_per_bit 
            bits((bit_no-1)*samples_per_bit+sample)=bit_pattern(bit_no);
        end
    end
    
    % See what it looks like
    figure;
    subplot(2,1,1);plot(t,bits);
    ylabel('Amplitude');
    title('Info signal');
    axis([0 tmax -2 2]);
    grid on;
    
    % PSK modulation
    PSK=[];
    if n==2
        for bit_no=1:1:length(bit_pattern)
            if bit_pattern(bit_no)==0
                t_bit = (bit_no- 1)*samples_per_bit*delt:delt:(bit_no*samples_per_bit-1)*delt;
                Wc = 2*pi*Cf*t_bit;
                mod = (1)*sin(Wc + 0);
            elseif bit_pattern(bit_no)==1
                t_bit = (bit_no- 1)*samples_per_bit*delt:delt:(bit_no*samples_per_bit-1)*delt;
                Wc = 2*pi*Cf*t_bit;
                mod = (1)*sin(Wc + pi);
            end
            PSK=[PSK mod];
        end
    elseif n==4
        for bit_no=2:2:length(bit_pattern)
            disp(bit_pattern(bit_no-1: bit_no));
            if bit_pattern(bit_no-1: bit_no)==[0, 0]
                t_bit = (bit_no-2)*samples_per_bit*delt:delt:((bit_no)*samples_per_bit-1)*delt;
                Wc = 2*pi*Cf*t_bit;
                mod = (1)*sin(Wc + 0);
            elseif bit_pattern(bit_no-1: bit_no)==[0, 1]
                t_bit = (bit_no-2)*samples_per_bit*delt:delt:((bit_no)*samples_per_bit-1)*delt;
                Wc = 2*pi*Cf*t_bit;
                mod = (1)*sin(Wc + pi/3);
            elseif bit_pattern(bit_no-1: bit_no)==[1, 1]
                t_bit = (bit_no-2)*samples_per_bit*delt:delt:((bit_no)*samples_per_bit-1)*delt;
                Wc = 2*pi*Cf*t_bit;
                mod = (1)*sin(Wc + 2*pi/3);
            else
                t_bit = (bit_no-2)*samples_per_bit*delt:delt:((bit_no)*samples_per_bit-1)*delt;
                Wc = 2*pi*Cf*t_bit;
                mod = (1)*sin(Wc + pi);
            end
            PSK=[PSK mod];
        end
    end
    subplot(2,1,2); plot(t,PSK);
    ylabel('Amplitude');
    title('ASK Modulated Signal');
    axis([0 tmax -2 2]);
    grid on;
end
