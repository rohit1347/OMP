I = imread('MONALISA_color.jpg');
I=rgb2gray(I);
I=imresize(I,[340,340]);

% Creating sparse matrix using DCT
J = dct2(I);

% Sparsifying DCT
J(abs(J) < 10) = 0;
sparse_image = idct2(J);

recon_image  = zeros(340,340);
sparse_image = J;

A_row = 50;                 %Set M (number of rows here). M must be <=N
M = A_row;
    for dct_col = 1:length(sparse_image)
        x = sparse_image(:,dct_col);
        N = length(sparse_image);
        
        % Random Dictionary
        A = randn(M,N);
        %Normalizing
        A = normc(A);
        y = A*x;
        % Store the basis which contribute the most to sparse elements, to A_new
        A_new = [];
        
        % r=residue
        r  = y;
        x_rec = zeros(N,1);
        max_index_array = zeros(N,1);
        for j=1:N
            % Find column of A that has the maximum projection on y
            w = A'*r;
            [~,sp_idx] = max(abs(w));
            x_rec(sp_idx,1) = 1;
            max_index_array(j,1) = sp_idx;
            % Add that column to A_basis
            A_new = [A_new, A(:,sp_idx)];
            l_p = pinv(A_new(:,1:j))*y;
            %new_r=New reidue
            new_r = y - A_new*l_p;
            residue_limit = norm(new_r);
            if j==N || residue_limit <= 1e-3
                x_rec(max_index_array(1:j,1),1) = l_p;
                recon_image(:,dct_col) = x_rec;
                norm_error=norm(x-x_rec)/norm(x);
                break;
            end
            r = new_r;
        end
    end

recon_image = idct2(recon_image);
figure
subplot(1,3,2)
imagesc(recon_image)
colormap('gray')
title(['Recovered Image M/N= ',num2str(A_row/length(J))]);
subplot(1,3,3)
% imshowpair(I,sparse_image,'montage')
imagesc(I)
colormap('gray')
title('Original Image')
subplot(1,3,1)
imagesc(idct2(sparse_image))
colormap('gray')
title('Sparsified Image (to recover)')
