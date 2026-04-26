function  y = resample_audio(y, f_old ,f_new ,n1)

[L, M] = find_resample_step(f_old, f_new);
new_length = L * length(y);
new_y = zeros(1, new_length);
index_y = 1;
little_count = 0;
for index = 1:new_length
    if (little_count == L)
        little_count = 0;
        new_y(index) = y(index_y);
        index_y = index_y + 1;
    end
    little_count = little_count + 1;
end
f_new1 = f_old * L;
f_old1 = f_old/2;
Ap = 0.1; 
Ast = 60; 
Wp = f_old1 / (f_new1/2); 
fst = f_old1 * 1.1;  
Ws = fst / (f_new1/2);
[g, Ws] = cheb2ord(Wp, Ws, Ap, Ast);
[b, a] = cheby2(g, Ast, Ws, 'low');   
y = filter(b, a, new_y);
y = y * L;
b = fir1(n1, 1/M, hamming(n1+1));
y = filter(b, 1, y);    
delay = n1/2;
y = y(delay+1:end);
y = y(1:M:end);
end