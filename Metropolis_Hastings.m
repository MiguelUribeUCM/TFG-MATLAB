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

% posterior gaussiano usando el teorema
Gamma_post_inv = (A_deblur' * A_deblur) / (sigma^2) + Gamma_pr_inv;
Gamma_post_inv = 0.5 * (Gamma_post_inv + Gamma_post_inv');

aux = (A_deblur' * g_noise) / (sigma^2);
post_mean = Gamma_post_inv \ aux;

% implementación de MH con random walk para varios valores de tau

fprintf('Muestreo MH en la malla blurred...\n');

% tamaño de la muestra
S = 1000;

% distintos valores de tau a probar
taus = [0.0060, 0.0040, 0.0025, 0.0010];
ntau = length(taus);

% guardar resultados
map_estimates = zeros(n_model, ntau);
cm_estimates = zeros(n_model, ntau);
acc_rates = zeros(ntau,1);

for itau = 1:ntau
    
    tau = taus(itau);
    fprintf('\ntau = %.4f\n', tau);
    
    % estado inicial
    xk = x_tikh;
    
    % log-posterior inicial
    logpi_xk = -0.5 * ((xk - post_mean)' * (Gamma_post_inv * (xk - post_mean)));
   
    accepted = 0;
    k = 0;
    
    % inicialización de estimadores MAP y CM
    cm_curr = zeros(n_model,1);
    map_curr = xk;
    logpi_best = logpi_xk;

    while k < S
        k = k + 1;

        % propuesta 
        y = xk + tau * randn(n_model,1);
        
        % log posterior
        logpi_y = -0.5 * ((y - post_mean)' * (Gamma_post_inv * (y - post_mean)));
        
        % log alpha
        logalpha = logpi_y - logpi_xk;
        
        % aceptar/rechazar
        if log(rand) <= min(0, logalpha)
            xk = y;
            logpi_xk = logpi_y;
            accepted = accepted + 1;
        end
        
        % actualizar estimadores
        cm_curr = cm_curr + xk;
        if logpi_xk > logpi_best
            logpi_best = logpi_xk;
            map_curr = xk;
        end
    end
    
    accrate = accepted / S;
    acc_rates(itau) = accrate;
    fprintf('Tasa de aceptación MH: %.7f\n', accrate);
    
    % guardar estimadores
    map_estimates(:,itau) = map_curr;
    cm_estimates(:,itau)  = cm_curr / S;
end

% visualización de resultados según el valor de tau 
figure('Name','MAP/CM para distintos valores de tau','Color','w');

for itau = 1:ntau
    map_tau = min(max(map_estimates(:,itau), 0), 1);
    map_tau  = reshape(map_tau, Ny_deblur, Nx_deblur);
    map_tau = max(map_tau(:)) - map_tau;
    
    subplot(2, ntau, itau);
    imagesc(map_tau); axis image; colormap gray; set(gca, 'XTick', [], 'YTick', []);
    title('MAP','Fontsize',21);
end


for itau = 1:ntau
    cm_tau = min(max(cm_estimates(:,itau), 0), 1);
    cm_tau  = reshape(cm_tau, Ny_deblur, Nx_deblur);
    cm_tau = max(cm_tau(:)) - cm_tau;
    
    subplot(2, ntau, ntau+itau);
    imagesc(cm_tau); axis image; colormap gray; set(gca, 'XTick', [], 'YTick', []);
    title('CM','Fontsize',21);
    xlabel(sprintf('\n\\tau = %.4f\ntasa = %.4f', taus(itau), acc_rates(itau)), 'FontSize', 21);
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