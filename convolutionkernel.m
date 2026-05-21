% dominio
x = -1:0.001:1;
y = -1:0.001:1;

% malla
[X, Y] = meshgrid(x, y);

% hypot para elemento a elemento
Z = exp(-20 * hypot(X, Y)); 

% graficar
figure('Color', 'w');
surf(X, Y, Z);

% estética
shading interp;       
colorbar;
colormap gray;
axis tight;

view(0,90);
axis equal;