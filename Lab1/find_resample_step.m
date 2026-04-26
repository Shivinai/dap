function [L, M] = find_resample_step(fs_old, fs_new)
    count = 0;
    max_count = 10;
    x = fs_new/fs_old;
    p2 = 0;
    p1 = 1;
    q2 = 1;
    q1 = 0;
    a = round(x);
    rem = x - a;
    while (count ~= max_count)
        p_curr = a*p1 + p2;
        q_curr = a*q1 + q2;
        if (rem == 0)
            L = abs(p_curr);
            M = abs(q_curr);
            break
        end
        x_new = 1/rem;
        a = round(x_new);
        rem = x_new - a;
        p2 = p1;
        p1 = p_curr;
        q2 = q1;
        q1 = q_curr;
        count = count + 1;
    end
end