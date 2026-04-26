function discret = ACP(y, N)
    q = 1:length(y);
    discret = q;
    ref = 1/(2^N);
    B = 0;
    for c = 1:length(y)
     
        for k = 1:N
            if abs(y(c)) > (2^(N-k) + B)*ref
            B = bitset(B, N - k + 1);
            end
        end
        discret(c) = B * sign(y(c));
        B = 0;
    end 
end