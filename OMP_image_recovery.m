close all
I = imread('MONALISA_color.jpg');
I=rgb2gray(I);
I=imresize(I,[340,340]);
%i = 5;

%image_block = image(10*i+1:10*(i+1),10*i+1:10*(i+1));
%image_block = double(image_block);
%imagesc(image_block);
%colormap('gray')

%image_block = reshape(image_block, [100,1]);


% Compute DCT
J = dct2(I);


% Sparse DCT
J(abs(J) < 50) = 0;
sparse_image = idct2(J);
% figure
% imshowpair(I,sparse_image,'montage')
% title('Original Grayscale Image (Left) and Processed Image (Right)');

recon_image  = zeros(340,340);
J_var=10;

%J is the sparsified matrix we have to reconstruct
residue_limit=1e-12;
for n_iter=1:10
for dct_col=1:length(J)
    for A_row=length(J)-J_var
        A=randn(A_row,length(J));
        A=normc(A);
        x=J(:,dct_col);
        r=A*x;
        
        A_new=[];
        y=A*x;
        x_rec_idx=zeros([A_row 1]);
        x_rec=zeros(size(x));
%         disp(sprintf('run= %d,Spars= %d , M= %d\n\n',run,support_length,M))
        maxrun=0;
        while(max(abs(r)>residue_limit)==1&& maxrun<=100)
            w=A'*r;
            [~,sp_idx]=max(abs(w));         %Taking absolute value of lambda
            A_new=[A_new A(:,sp_idx)];
            l_p=pinv(A_new)*y;
            x_rec_idx(sp_idx)=1;            %Storing the indices where sparse elements are present
            r=y-A_new*l_p;
            maxrun=maxrun+1;
        end
        for j_iter=1:length(l_p)
            flag=0;
            for k_iter=1:length(x_rec_idx)
                if x_rec_idx(k_iter)==1 && flag==0
                    x_rec(k_iter)=l_p(j_iter);
                    x_rec_idx(k_iter)=0;
                    flag=flag+1;
                end
            end
        end
        recon_image(:,dct_col)=x_rec/n_iter;
    end
end
end
recon_image=idct2(recon_image);
figure
subplot(1,3,2)
imagesc(recon_image)
colormap('gray')
title(['Recovered Image M/N= ',num2str((length(J)-J_var)/length(J))]);
subplot(1,3,3)
% imshowpair(I,sparse_image,'montage')
imagesc(I)
colormap('gray')
title('Original Image')
subplot(1,3,1)
imagesc(sparse_image)
colormap('gray')
title('Sparsified Image (to recover)')