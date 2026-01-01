%%% RECONAISSANCE FACIALE

%% Informations générales

% Auteur : Rayan Kobrossly
% TD7 IA2R
% ------------------------------


clear
clc
close all;
load Yalefaces.mat
load baseunknown.mat
% size_Xtrain = size(X_train);

%% Rendre le vecteur colonne en une matrice 64x64

for i=1:90
    cat_train{i} = mat2gray(imageto64(X_train(:,i))); %le fait en une seule ligne de code
    % uint8 et mat2gray transforme le double en un entier positif entre 0
    % et 255
end


%% Représentation des images 64x64
figure;
titres = {'Happy', 'Glasses', 'Sad', 'Sleepy', 'Surprised', 'No Glasses', 'Normal'};
for i=1:7
    subplot(2,4, i); 
    imshow(cat_train{i});
    title(titres{i});
end


%% Calcul du visage moyen
xmoy = mean(X_train,2); %calcul le visage moyen de toutes les 90 photos
centre  = mat2gray(imageto64(xmoy)); %%transforme le vecteur colonne 4096x1 en 64x64
figure;
plot(centre);
imshow(centre);

X_c = X_train - xmoy; %calcul image centré

figure(10);
subplot(1,5,1); % affiche visage centre
imshow(centre);
title("Image visage centrée");

subplot(1,5,2); % affiche visage n1 non centrée
imshow(cat_train{1});
title("Image n1 non centrée");

subplot(1,5,3); % affiche visage n1 centrée
imshow(uint8(imageto64(X_c(:,1))));
title("Image n1 centrée");

subplot(1,5,4); % affiche visage n1 non centrée
imshow(cat_train{50});
title("Image n2 non centrée");

subplot(1,5,5); % affiche visage n1 centrée
imshow(uint8(imageto64(X_c(:,50))));
title("Image n2 centrée");

%% Calcul des caractéristiques d'un visage
[U, S, V] = svd(X_c, 0);
orthonormality_U = norm(U' * U - eye(size(U, 2)));  % Devrait être quasi égale à 0

S_carre = diag(S).^2; % calcul le carre de chaque valeur, la matrice diagonale est mise en un vecteur colonne
sum_S_carre = sum(S_carre);
p_S_carre = S_carre/sum_S_carre; % pourcentage de valeurs propres


sum_p_S_carre = zeros(90,1);
sum_p_S_carre(1,1) = p_S_carre(1,1);
 for k=2:90
     sum_p_S_carre(k,1) = sum_p_S_carre(k-1,1) + p_S_carre(k,1);
 end


v_k = (1:1:90)'; %pour tracer la courbe de somme des pourcentage des valeurs propres en fonction de nombre d'elements
figure(11);
plot(v_k,sum_p_S_carre);
title("La somme des pourcentage des valeurs propores en fonction du nombre d'élements");
xlabel("L'ensemble des élements");
ylabel("La somme des pourcentages");
% au bout de 15 images, le pourcentage des valeurs propores est de 80.7%
% pour avoir 75% des données, il faut avoir recours à seulement 11 éléments


%% Projection dans le sous espace des visages
indice_x =1;
j = 2;
liste_k={5;15;25;35;50;60;70;80;90};
for i=1:length(liste_k) %tester pour les differentes valeurs de k
    k = liste_k{i};
x = X_train(:,indice_x); %prenons comme exemple la premiere image
z = U(:,1:k)' * X_c(:,indice_x); %x-xmoyenne represente l'image centree de x
% il y a pas le . avant * car c'est une multiplication matricielle non pas
% terme à terme
x_reconstruit = xmoy + U(:,1:k)*z; %l'image reconstruite de x
%U(:,k).*z doit etre 4096x1 non pas 4096*4096

e = norm(x-x_reconstruit); %l'erreur entre l'image et on image reconstruite
fprintf('error is equal to %.10f\n', e);


figure(53);
subplot(2,5,1); % image origine
imshow(mat2gray(imageto64(x)));
title("Image 1");
subplot(2,5,j); % image origine
imshow(mat2gray(imageto64(x_reconstruit)));
title("k="+k);
j =j+1;

end
clear j
clear i
clear k
clear indice_x

%% Identification d'un visage

%********************************
 liste_k={5;30;50;75;90};
for ii=1:length(liste_k) %tester pour les differentes valeurs de k
    k = liste_k{ii};
   indice_subplot =1;

for indice_test=1:30 %pour toutes les personnes de X_test

e_k_distance = zeros(1, k); %on initialise la distance e_k_distance
zmoy = mean(X_test,2);
Z_c = X_test-zmoy;
z_test = U(:,1:k)' * Z_c(:,indice_test); %l'image de la personne de la matrice test
for i=1:90
z_train = U(:,1:k)' * X_c(:,i); % l'un des 90 personnes de la base train pour pouvoir calculer la distance entre les personnes
e_k_distance(i) = norm(z_test-z_train); %les differents elements du vecteur distance 1xk = 1x90

end 

[min_e_k, indice_e_k] = min(e_k_distance); %calcul de la distance minimale= personne la plus probable d'etre bien identifiée
fprintf('Minimum value: %.4f at index: %d\n', min_e_k, indice_e_k);
indice_individu_test = id_train(indice_e_k);
fprintf('the individual %d id is equal to %d\n',indice_test, indice_individu_test);

 
%*******************************
    figure(12+ii) % on aura plusieurs figures pour les differentes valeurs de k
    subplot(6,10,indice_subplot); % image origine
    imshow(mat2gray(imageto64(X_test(:,indice_test)))); %trace l'image de X_test
    title("Image n "+indice_test);
    indice_subplot = indice_subplot+1;
    subplot(6,10,indice_subplot); % trace l'image de la personne la plus proche
    imshow(mat2gray(imageto64(X_train(:,indice_e_k)))); 
    title("Train id "+id_train(indice_e_k));
    indice_subplot = indice_subplot+1;
    
    sgtitle("For k = " + string(k));

end

end


%% Pour aller plus loin

k = 30;

%pour l'erreur minimale de la base d'entrainement pour k=30

erreur_train = zeros(1, 90);
xmoy = mean(X_train,2);
X_c = X_train - xmoy; %calcul image centré

for indice_x=1:90  %calculer l'erreur pour toutes les personnes
x = X_train(:,indice_x); %prenons comme exemple la premiere image
z = U(:,1:k)' * X_c(:,indice_x); %x-xmoyenne represente l'image centree de x
x_reconstruit = xmoy + U(:,1:k)*z; %l'image reconstruite de x


erreur_train(indice_x) = norm(x-x_reconstruit); %l'erreur entre l'image et on image reconstruite

end

max_erreur_train = max(erreur_train);
moy_erreur_train = mean(erreur_train);
fprintf('the maximale de la base train %d et la moyenne est égale à %d\n',max_erreur_train, moy_erreur_train);

figure(30);
subplot(1,2,1); % image origine
imshow(mat2gray(imageto64(x)));
title("Image 1");
subplot(1,2,2); % image reconstruite pour la dernier image de X_train par exemple
imshow(mat2gray(imageto64(x_reconstruit)));
title("Reconstruction train");

%******************************
%pour l'erreur minimale de la base de test pour k=30
erreur_test = zeros(1, 30);
xmoy_test = mean(X_train,2);
X_c_test = X_test - xmoy_test; %calcul image centré
for indice_x=1:30  %calculer l'erreur pour toutes les personnes
x_test = X_test(:,indice_x); %prenons comme exemple la premiere image
z = U(:,1:k)' * X_c_test(:,indice_x); %x-xmoyenne represente l'image centree de x

x_reconstruit_test = xmoy_test + U(:,1:k)*z; %l'image reconstruite de x


erreur_test(indice_x) = norm(x_test-x_reconstruit_test); %l'erreur entre l'image et on image reconstruite

end
max_erreur_test = max(erreur_test);
moy_erreur_test = mean(erreur_test);
fprintf('the maximale de la base test %d et la moyenne est égale à %d\n',max_erreur_test, moy_erreur_test);

figure(31);
subplot(1,2,1); % image origine
imshow(mat2gray(imageto64(x_test)));
title("Image 1");
subplot(1,2,2); % image reconstruite pour la dernier image de X_test par exemple
imshow(mat2gray(imageto64(x_reconstruit_test)));
title("Reconstruction train");

%******************************
%pour l'erreur minimale de la base no visage pour k=30
erreur_nf = zeros(1, 10);
xmoy_nf = mean(X_noface,2);
X_c_nf = X_noface - xmoy_nf; %calcul image centré
for indice_x=1:10  %calculer l'erreur pour toutes les personnes
x_nf = X_noface(:,indice_x); %prenons comme exemple la premiere image
z = U(:,1:k)' * X_c_nf(:,indice_x); %x-xmoyenne represente l'image centree de x

x_reconstruit_nf = xmoy_nf + U(:,1:k)*z; %l'image reconstruite de x


erreur_nf(indice_x) = norm(x_nf-x_reconstruit_nf); %l'erreur entre l'image et on image reconstruite

end
min_erreur_nf = min(erreur_nf); % calcul la valeur minimale
moy_erreur_nf = mean(erreur_nf);
fprintf('the minimum de la base no face %d et la moyenne est égale à %d\n',min_erreur_nf, moy_erreur_nf);

figure(32);
subplot(1,2,1); % image origine
imshow(mat2gray(imageto64(x_nf)));
title("Image 1");
subplot(1,2,2); % image reconstruite pour la dernier image de X_test par exemple
imshow(mat2gray(imageto64(x_reconstruit_nf)));
title("Reconstruction train");

%**********************************************
%**********************************************
% Algorithme de face/noface

disp("Voici les images qu'on doit classifier");
figure(33);
for i=1:10
    cat_noface{i} = mat2gray(imageto64(X_inconnu(:,i)));
    subplot(2,5, i); 
    imshow(cat_noface{i});
    title("Image "+i);
  
end

erreur_face = mean([max_erreur_train,max_erreur_test,moy_erreur_test]) + 0.5*mean([moy_erreur_train,moy_erreur_test]);
% l'erreur moyenne d'un visage que ca soit entrainement ou train
%la formule que j'ai trouvé marche le plus
erreur_inc = zeros(1, 10);
xmoy_inc = mean(X_inconnu,2);
X_c_inc = X_inconnu - xmoy_inc; %calcul image centré
for indice_x=1:10  %calculer l'erreur pour toutes les personnes
x_inc = X_inconnu(:,indice_x); %prenons comme exemple la premiere image
z = U(:,1:k)' * X_c_inc(:,indice_x); %x-xmoyenne represente l'image centree de x

x_reconstruit_inc = xmoy_inc + U(:,1:k)*z; %l'image reconstruite de x


erreur_inc(indice_x) = norm(x_inc-x_reconstruit_inc); %l'erreur entre l'image et on image reconstruite
if erreur_inc(indice_x)<=erreur_face
    fprintf('the image avec le numero %d est un visage\n',indice_x);
else
    fprintf('the image avec le numero %d n est pas un visage\n',indice_x);
end

end

