clc
close all
clear all

%Experimental setup

%-----------------
%Control Variables pt1

n_iter=20;
N_set=[20,50,100];
num_fig=1;
n_sigma_set=[1e-3,0.1];

%---------------------------Noisy case begins here-------------------------

sparsity_flag_set=[0,1];
%______________________________________________________________________
% sparsity_flag=0 cCORRESPONDS TO SPARSITY KNOWN CASE
% sparsity_flag=1 cCORRESPONDS TO SPARSITY UNKNOWN CASE
%______________________________________________________________________
for n_sigma=n_sigma_set
    for N=N_set
        %-----------------
        %Control Variables pt2
        M_lim=ceil(0.75*N);
        residue_limit=1e-6;
        noisy_residue_limit=residue_limit*100;
        success_limit=0.001;
        noisy_success_limit=1*success_limit;
        mu_A=0.4;
        sigma_A=1;
        n_mean=0;
        n_sigma_set=[1e-3,0.1];
        smax=floor(N/2);
        
        %-----------------
        for sparsity_flag=sparsity_flag_set
            norm_error_noisy=zeros(M_lim,smax);
            esr_noisy=zeros(M_lim,smax);
            for run=1:n_iter
                for M=1:M_lim
                    for support_length=1:smax
                        %Creating A with dimensions M=5, N=100
                        A=mu_A + sigma_A*randn(M,N);
                        normalize=sqrt(sum(A.^2,1));
                        [mdim_A,ndim_A]=size(A);
                        for i=1:ndim_A
                            A(:,i)=A(:,i)./normalize(i);             %Normalizing the columns of A
                        end
                        
                        %Begin generation of x
                        support_A=randi([1 N],[1 support_length]);   %Described as S in the question
                        sp1_length=randi([1 length(support_A)]);   %Number of elements lying in the range[-10,-1] are chosen randomly
                        sp2_length=length(support_A)-sp1_length;   %Number of elements lying in the range [1,10] = length(Support)-Number of elements lying in the range[-10,-1]
                        sp1=-10+9.*rand(1,sp1_length);              %Elements in the range[-10,-1]
                        sp2=1+9.*rand(1,sp2_length);                %Elements in the range[1,10]
                        sparse_elements=[sp1 sp2];
                        x=zeros(1,N);
                        idx_count=1;
                        for i=support_A
                            x(i)=sparse_elements(idx_count);
                            idx_count=idx_count+1;
                        end
                        x=x';                               %X has dimensions Nx1
                        %End generation of x
                        
                        %Begin generation of noise
                        n=n_mean + n_sigma.*randn([M 1]);
                        %End generation of noise
                        
                        %-----------------------------SPARSE RECOVERY-----------------------------%
                        r=A*x+n;
                        A_new=[];
                        y=A*x+n;
                        x_rec_idx=zeros([N 1]);
                        max_index_array=[];
                        x_rec=zeros(size(x));
                        disp(sprintf('Noisy run= %d , Spars= %d , M= %d, Sig=%d, Flag=%d',run,support_length,M,n_sigma,sparsity_flag))
                        maxrun=0;
                        min_rec_flag=0;
                        %             while(max(abs(r)>residue_limit)==1 && maxrun<=100)
                        if sparsity_flag==0     %Sparsity known case
                            while(max(abs(r)>noisy_residue_limit)==1 && nnz(x)>nnz(x_rec_idx)&& maxrun<=100)
                                w=A'*r;
                                [~,sp_idx]=max(abs(w));         %Taking absolute value of lambda
                                max_index_array=[max_index_array sp_idx];
                                A_new=[A_new A(:,sp_idx)];
                                l_p=pinv(A_new)*y;
                                x_rec_idx(sp_idx)=1;            %Storing the indices where sparse elements are present
                                r=y-A_new*l_p;
                                %                                 A(:,sp_idx)=[];
                                maxrun=maxrun+1;
                            end
                            x_rec(max_index_array)=l_p;
                            norm_error_noisy(M,support_length)=norm_error_noisy(M,support_length)+(norm(x-x_rec)/norm(x)/n_iter);
                            %----------------Condition for recovery------------------%
                            if norm(x-x_rec)/norm(x)<=noisy_success_limit
                                esr_noisy(M,support_length)=esr_noisy(M,support_length)+1/n_iter;
                            end
                        else %Sparsity unknown case
                            maxrun=0;
                            max_index_array=[];
                            while(max(abs(r)>norm(n))==1&& maxrun<=100 &&min_rec_flag==0)
                                w=A'*r;
                                [~,sp_idx]=max(abs(w));         %Taking absolute value of lambda
                                max_index_array=[max_index_array sp_idx];
                                A_new=[A_new A(:,sp_idx)];
                                l_p=pinv(A_new)*y;
                                x_rec_idx(sp_idx)=1;            %Storing the indices where sparse elements are present
                                r=y-A_new*l_p;
                                min_rec_flag=min_rec_flag+1;
                                maxrun=maxrun+1;
                            end
                            x_rec(max_index_array)=l_p;
                            norm_error_noisy(M,support_length)=norm_error_noisy(M,support_length)+(norm(x-x_rec)/norm(x)/n_iter);
                            %----------------Condition for recovery------------------%
                            if norm(x-x_rec)/norm(x)<=(noisy_success_limit)
                                esr_noisy(M,support_length)=esr_noisy(M,support_length)+1/n_iter;
                            end
                        end
                    end
                    
                end
            end
            figure(num_fig)
            imagesc(esr_noisy)
            colormap('gray')
            title(['Probability of ESR- Noisy case - N=',num2str(N),' Sigma=',num2str(n_sigma), ' Sparsity flag=',num2str(sparsity_flag)])
            xlabel('Sparsity')
            ylabel('M rows')
            colorbar
            num_fig=num_fig+1;
            
            figure(num_fig)
            imagesc(norm_error_noisy)
            title(['Average Normalized Error- Noisy case - N=',num2str(N),' Sigma=',num2str(n_sigma),' Sparsity flag=',num2str(sparsity_flag)])
            colormap('gray')
            xlabel('Sparsity')
            ylabel('M rows')
            colorbar
            num_fig=num_fig+1;
        end
    end
end