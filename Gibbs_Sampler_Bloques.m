clear; clc; close all;

% semilla
rng(0)

% cargar imagen
imagen_rgb = imread('pxArt.png');

% poner en escala de grises
if size(imagen_rgb, 3) == 3, imagen_rgb = rgb2gray(imagen_rgb); end
f_bw = im2double(imagen_rgb);

% invertimos la imagen para que tenga bordes negros
% luego tenemos que invertir todas para visualizarlas
f_bw = max(f_bw(:)) - f_bw;

% esto es porque la imagen original tiene dos bandas de un color no blanco
f_bw(1:7,:) = 0;
f_bw(63:70,:) = 0;

[rows_orig, cols_orig] = size(f_bw);
aspect_ratio = cols_orig / rows_orig;

% mallas y tal
Ny_real = 70;
Nx_real = round(Ny_real * aspect_ratio);
h_real = 1/Ny_real;

% malla de la deblurred
Ny_deblur = 90;
Nx_deblur = round(Ny_deblur * aspect_ratio);
h_deblur = 1/Ny_deblur;

% meshgrid permite crear las mallas (real/deblur)
[X_real, Y_real] = meshgrid((0.5:Nx_real-0.5)*h_real, (0.5:Ny_real-0.5)*h_real);
p_real = [X_real(:), Y_real(:)];

[X_deblur, Y_deblur] = meshgrid((0.5:Nx_deblur-0.5)*h_deblur, (0.5:Ny_deblur-0.5)*h_deblur);
p_deblur = [X_deblur(:), Y_deblur(:)];

% núcleo de convolución para generar la imagen blurred
alpha_real = 20; 
kernel_real = @(r) exp(-alpha_real * r);

% dimensiones de A_grande (la matriz de blurring)
m = Ny_deblur * Nx_deblur;   
n = Ny_real * Nx_real;      
A_grande = zeros(m, n);
for i = 1:m
    dists = sqrt((p_deblur(i,1) - p_real(:,1)).^2 + (p_deblur(i,2) - p_real(:,2)).^2);
    A_grande(i,:) = (h_real^2) * kernel_real(dists)';
end

g_blurred = A_grande * f_bw(:);

% ponerle ruido de 1.5%
sigma = 0.015 * max(g_blurred);
noise = sigma * randn(size(g_blurred));
max_err = norm(noise);

g_noise = g_blurred + noise;

% tikhonov para alpha = 20

% crear A_deblur
alpha_model = 20;
kernel_model = @(r) exp(-alpha_model * r);

n_model = Ny_deblur * Nx_deblur;  
A_deblur = zeros(n_model, n_model);

for i = 1:n_model
    dists = sqrt((p_deblur(i,1) - p_deblur(:,1)).^2 + (p_deblur(i,2) - p_deblur(:,2)).^2);
    A_deblur(i, :) = (h_deblur^2) * kernel_model(dists)';
end

AtA = A_deblur' * A_deblur;
Atg = A_deblur' * g_noise;
I = eye(n_model);

% buscar delta aproximado (método de newton)
delta_opt = 1e-2;

for iter = 1:200
    M = AtA + delta_opt * I;
    x_curr = M \ Atg;

    resid_sq = sum((A_deblur * x_curr - g_noise).^2);
    f_val = resid_sq - max_err^2;

    if abs(f_val) < 1e-7 * max_err^2
        break
    end

    z = M \ x_curr;
    f_prime = 2 * delta_opt * (x_curr' * z);

    if f_prime < 1e-20
        f_prime = 1e-20;
    end

    delta_opt = delta_opt - f_val / f_prime;

    if delta_opt <= 0
        delta_opt = 1e-9;
    end
end

x_tikh = (AtA + delta_opt * I) \ Atg;

fprintf('Terminado Tikhonov: delta_opt = %.6e\n', delta_opt);

% prior gaussiano
lambda_prior_blur = 10;

L_blur = build_laplacian_2d(Ny_deblur, Nx_deblur);

Gamma_pr_inv = lambda_prior_blur * L_blur;
Gamma_pr_inv = 0.5 * (Gamma_pr_inv + Gamma_pr_inv');

% posterior gaussiano
Gamma_post_inv = (A_deblur' * A_deblur) / (sigma^2) + Gamma_pr_inv;
Gamma_post_inv = 0.5 * (Gamma_post_inv + Gamma_post_inv');

aux = (A_deblur' * g_noise) / (sigma^2);

post_mean = Gamma_post_inv \ aux;

% implementación de gibbs sampler por bloques

fprintf('Muestreo GS en la malla blurred...\n');

% tamaño de la muestra
Nsweeps = 1000;

% tamaños de bloque a probar
block_sizes = [1, 30, 67, 90];
nblocksizes = length(block_sizes);

% guardar estimadores
map_blocks = zeros(n_model, nblocksizes);
cm_blocks = zeros(n_model, nblocksizes);

for isize = 1:nblocksizes
    
    blockdim = block_sizes(isize);

    fprintf('\nTamaño de bloque = %d\n', blockdim);

    Nblocks = n_model / blockdim;

    xk = x_tikh;
    x_cm = zeros(n_model,1);
    logmap = -Inf;
    x_map = xk;

    for sweeps = 1:Nsweeps
        
        for iblock = 1:Nblocks
            % índices correspondientes al bloque (es horizontal)
            B = (blockdim*(iblock-1)+1):blockdim*iblock;
    
            % índices complementarios del bloque
            Bc = [1:blockdim*(iblock-1), (blockdim*iblock+1):n_model];
    
            % algoritmo muestreo condicional de la referencia [7]
            mu_B = post_mean(B);
            mu_Bc = post_mean(Bc);
    
            Gamma_post_inv_BB = Gamma_post_inv(B,B);
            Gamma_post_inv_BBc = Gamma_post_inv(B,Bc);
    
            aux = Gamma_post_inv_BBc * (xk(Bc) - mu_Bc);
            mu_B_cond = mu_B - (Gamma_post_inv_BB \ aux);
    
            Lchol = chol(Gamma_post_inv_BB, 'upper');
    
            z = randn(blockdim,1);
            v = Lchol \ z;
            
            % fin del muestreo condicional
            xk(B) = mu_B_cond + v;
        end
        
        % actualizar los estimadores
        x_cm = x_cm + xk;
        logpost = -0.5 / sigma^2 * norm(A_deblur * xk - g_noise)^2 - 0.5 * (xk' * Gamma_pr_inv * xk);
        if logpost > logmap
            logmap = logpost;
            x_map = xk;
        end
    end
    map_blocks(:, isize) = x_map;
    cm_blocks(:, isize) = x_cm / Nsweeps;
end

% visualización de resultados según el tamaño del bloque
figure('Name','MAP/CM para distintos tamaños de bloque','Color','w');

% fila superior: MAP
for isize = 1:nblocksizes
    map_gs = min(max(map_blocks(:,isize), 0), 1);
    map_gs = reshape(map_gs, Ny_deblur, Nx_deblur);
    map_gs = max(map_gs(:)) - map_gs;

    subplot(2, nblocksizes, isize);
    imagesc(map_gs); axis image; colormap gray; set(gca, 'XTick', [], 'YTick', []);
    title('MAP','FontSize',21);
end

% fila inferior: CM
for isize = 1:nblocksizes
    cm_gs = min(max(cm_blocks(:,isize), 0), 1);
    cm_gs = reshape(cm_gs, Ny_deblur, Nx_deblur);
    cm_gs = max(cm_gs(:)) - cm_gs;

    subplot(2, nblocksizes, nblocksizes + isize);
    imagesc(cm_gs); axis image; colormap gray; set(gca, 'XTick', [], 'YTick', []);
    title('CM','FontSize',21);
    xlabel(sprintf('card = %d', block_sizes(isize)), 'FontSize', 21);
end

% función para crear la discretización del laplaciano en una malla 2D
function L = build_laplacian_2d(Ny, Nx)

    n = Ny * Nx;
    L = zeros(n, n);

    for i = 1:Ny
        for j = 1:Nx
            
            % índice lineal del píxel (i,j)
            k = sub2ind([Ny, Nx], i, j);
            
            L(k,k) = 4;
            
            % vecinos de arriba, abajo, izquierda y derecha
            if i > 1
                kup = sub2ind([Ny, Nx], i-1, j);
                L(k,kup) = -1;
            end
            
            if i < Ny
                kdown = sub2ind([Ny, Nx], i+1, j);
                L(k,kdown) = -1;
            end
            
            if j > 1
                kleft = sub2ind([Ny, Nx], i, j-1);
                L(k,kleft) = -1;
            end
            
            if j < Nx
                kright = sub2ind([Ny, Nx], i, j+1);
                L(k,kright) = -1;
            end
        end
    end
end

