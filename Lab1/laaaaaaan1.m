n = 1:200;
y = -sin(7*pi/100*n) - cos(pi*n/125 - pi/4);z
y = y/max(abs(y));

N = 10;
y_cast = ACP(y, N);

y_supercast = y_cast/abs(max(y_cast));
figure
hold on
stairs(n, y_supercast, 'b')
stem(n, y, 'r')
xlabel('номер отсчёта')
ylabel('нормированные амплитуды сигналов')
hold off
figure
errr = y - y_supercast;
stem(n, errr, 'b')
xlabel('номер отсчёта')
ylabel('значение ошибки')
f_old = 8000; 
f_new = 14000;
[x, JustF] = audioread("ML70_06.wav");
[P, Q] = rat(f_old / JustF); 
y = resample(x, P, Q);
SIGNAL = resample(x, P, Q);
xlabel('номер отсчёта')
ylabel('значение сигнала')
%%%
n = 24;
y = resample_audio(y, f_old, f_new, n);
q = 1:length(y);
o = 1:length(SIGNAL);

figure
plot(q, y);
xlabel('номер отсчёта')
ylabel('значение передескретизированного сигнала')
figure
specgram(SIGNAL,512,f_old,hann(512),475);
set(gca,'Clim', [-80 5])
figure
specgram(y,512,f_new,hann(512),475);
set(gca,'Clim', [-80 5])

%
[P, Q] = rat(f_new / f_old); 
TempY = resample(SIGNAL, P, Q);
temp = 1:length(TempY);
y(end+1:end+3) = 0;
[corr, lag] = xcorr(TempY, y, 'none');
[~, max_idx] = max(abs(corr));
shift_samples = lag(max_idx);
TempY_shifted = zeros(length(TempY) + 5, 1);
TempY_shifted(6:end) = TempY;
[corr, lag] = xcorr(TempY_shifted, y, 'none');
[~, max_idx] = max(abs(corr));
shift_samples = lag(max_idx);
disp(shift_samples);
ER = 1:length(TempY);
for i = 1 : length(TempY)
    ER(i) = (TempY_shifted(i) - y(i))^2;
end
figure
hold on
w = 1:length(TempY_shifted);
e = 1:length(y);
stem(w,TempY_shifted)
stem(e, y)
hold off

index = 1 : length(ER);
figure
stem(index, ER)
xlabel('номер отсчёта')
ylabel('значение ошибки')
