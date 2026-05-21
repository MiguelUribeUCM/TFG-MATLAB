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

% plot de las imágenes real y blurred
figure('Name', 'Directo: Original vs Borroso', 'Position', [100, 450, 1200, 380]);
colormap gray;

f_true_plot = max(f_bw(:)) - f_bw;
subplot(1,2,1);
imagesc(f_true_plot); axis image; 
title('');
set(gca, 'XTick', [], 'YTick', []);

g_img = reshape(g_noise, [Ny_deblur, Nx_deblur]);
g_img = min(max(g_img, 0), 1);
g_plot = max(g_img(:)) - g_img;
subplot(1,2,2);
imagesc(g_plot); axis image;
title('');
set(gca, 'XTick', [], 'YTick', []);

% figure para plot de los 8 casos
figure('Name', 'tikhonov', 'Position', [50, 50, 1400, 700]);
colormap gray;

% 8 alphas para probar 
alphas_prueba = [3, 11, 18, 20, 22, 30, 45, 60]; 
fprintf('bucle (8 casos)\n');

% bucle alphas
for i = 1:length(alphas_prueba)
    
    fprintf('  (%d/%d): alpha = %d \n', i, length(alphas_prueba), alphas_prueba(i));
    
    % cambiar los núcleos y A_deblur para todo alpha
    kernel_model = @(r) exp(-alphas_prueba(i) * r);
    n_model = Ny_deblur * Nx_deblur;
    A_deblur = zeros(n_model, n_model);
    
    % construcción de matriz
    for j = 1:n_model
        dists = sqrt((p_deblur(j,1) - p_deblur(:,1)).^2 + (p_deblur(j,2) - p_deblur(:,2)).^2);
        A_deblur(j, :) = (h_deblur^2) * kernel_model(dists)';
    end
    
    % tikhonov
    AtA = A_deblur' * A_deblur;
    Atg = A_deblur' * g_noise;
    I = eye(n_model);
    
    % buscar delta aproximado (método de newton)
    delta_opt = 1e-2;
    
    for iter = 1:500
        M = AtA + delta_opt * I;
        x_delta = M \ Atg;
        
        resid_sq = sum((A_deblur * x_delta - g_noise).^2);
        f_val = resid_sq - max_err^2;
        
        if abs(f_val) < 1e-7 * max_err^2
            break; 
        end
        
        z = M \ x_delta;
        f_prime = 2 * delta_opt * (x_delta' * z);
        
        if f_prime < 1e-20
            f_prime = 1e-20; 
        end

        delta_opt = delta_opt - f_val / f_prime;
        
        if delta_opt <= 0
            delta_opt = 1e-9; 
        end
    end

    % imagen final
    f_final = reshape((AtA + delta_opt*I) \ Atg, [Ny_deblur, Nx_deblur]);
    f_final = min(max(f_final, 0), 1);
    f_final = max(f_final(:)) - f_final;
    
    % plot
    subplot(2, 4, i);
    imagesc(f_final); axis image;  
    set(gca, 'XTick', [], 'YTick', []);
    xlabel(['\alpha=' num2str(alphas_prueba(i)), ', \delta=' num2str(delta_opt, '%.1e')],'FontSize', 21);
end
