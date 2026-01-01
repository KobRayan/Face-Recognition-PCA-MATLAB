function matrix_square = imageto64(matrix_ligne)

matrix_square = zeros(64,64);
if size(matrix_ligne)~=[4096,1]

fprintf('Error, matrix size doesnt match\n');
else
    n=1;
    for k=1:4096
    if mod(k,64)==1 && k~=1
       n=n+1;
    end
    j = k -(n-1)*64;
    matrix_square(j,n) = matrix_ligne(k);
     
    end
end
