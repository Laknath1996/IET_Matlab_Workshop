Hfunction Hd = getFilter
%GETFILTER Returns a discrete-time filter object.

% MATLAB Code
% Generated by MATLAB(R) 9.4 and DSP System Toolbox 9.6.
% Generated on: 23-Jun-2020 10:28:22

% Butterworth Lowpass filter designed using FDESIGN.LOWPASS.

% All frequency values are in Hz.
Fs = 500;  % Sampling Frequency

Fpass = 100;         % Passband Frequency
Fstop = 120;         % Stopband Frequency
Apass = 3;           % Passband Ripple (dB)
Astop = 60;          % Stopband Attenuation (dB)
match = 'passband';  % Band to match exactly

% Construct an FDESIGN object and call its BUTTER method.
h  = fdesign.lowpass(Fpass, Fstop, Apass, Astop, Fs);
Hd = design(h, 'butter', 'MatchExactly', match);

% [EOF]