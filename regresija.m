pkg load statistics;

data = csvread('SeoulBikeData.csv');
X_full = data(:, [3 4 5 6 7 8 9 10 11]);
y_full = data(:, 2);
m = size(X_full, 1);
idx = randperm(m);
train_idx = idx(1:round(0.8 * m));
test_idx = idx(round(0.8 * m) + 1:end);

X_train_raw = X_full(train_idx, :);
y_train_raw = y_full(train_idx);
X_test_raw = X_full(test_idx, :);
y_test = y_full(test_idx);

[beta_final, optimalne_kolone, z] = backwards_stepwise(X_train_raw, y_train_raw);
X_train_sel = X_train_raw(:, optimalne_kolone);
X_test_sel = X_test_raw(:, optimalne_kolone);
X_full_sel = X_full(:, optimalne_kolone);

% linearna regresija i st podataka
[X_std, mu_X, sigma_X] = zscore(X_full_sel);
y_mean = mean(y_full);
y_std = std(y_full);
y_stdzd = (y_full - y_mean) / y_std;

X_std = [ones(size(X_std, 1), 1), X_std];
beta = (X_std' * X_std) \ (X_std' * y_stdzd);
y_pred_stdzd = X_std * beta;
y_pred = y_pred_stdzd * y_std + y_mean;

mse = mean((y_full - y_pred).^2);
fprintf("Linearna regresija MSE: %.4f\n", mse);

figure;
plot(y_full, y_pred, 'b.');
hold on;
plot([min(y_full), max(y_full)], [min(y_full), max(y_full)], 'r--');
xlabel('Stvarne vrednosti');
ylabel('Predikcija');
title('Linearna regresija');
grid on;
legend('Predikcije', 'Idealno');

% k-fold validacija
X = X_full_sel;
y = y_full;

N = size(X, 1);
K = 5;
part = cvpartition(N, 'KFold', K);
part = struct(part);
mse_folds = [];

for i = 1:K
    idx_train = (part.inds != i);
    idx_val   = (part.inds == i);

    X_train = X(idx_train,:);
    y_train = y(idx_train);
    X_val = X(idx_val,:);
    y_val = y(idx_val);

    [X_train_z, mu, sigma] = zscore(X_train);
    X_val_z = (X_val - mu) ./ sigma;
    y_mean = mean(y_train);
    y_std = std(y_train);
    y_train_z = (y_train - y_mean) / y_std;

    X_train_z = [ones(size(X_train_z,1),1), X_train_z];
    X_val_z = [ones(size(X_val_z,1),1), X_val_z];

    beta = (X_train_z' * X_train_z) \ (X_train_z' * y_train_z);
    y_val_pred_z = X_val_z * beta;

    y_val_pred = y_val_pred_z * y_std + y_mean;
    mse = mean((y_val - y_val_pred).^2);
    mse_folds = [mse_folds, mse];
end

cv_mse_mean = mean(mse_folds);
fprintf("Prosečan K-fold MSE: %.4f\n", cv_mse_mean);

figure;
bar(mse_folds);
xlabel('Fold');
ylabel('MSE');
title('K-fold validacija - Linearna regresija');
grid on;

% Ridge & Lasso regresija
mu_X = mean(X_train_sel);
sigma_X = std(X_train_sel);
X_train = (X_train_sel - mu_X) ./ sigma_X;
X_test = (X_test_sel - mu_X) ./ sigma_X;

y_mean = mean(y_train_raw);
y_std = std(y_train_raw);
y_train_std = (y_train_raw - y_mean) / y_std;

X_train = [ones(size(X_train, 1), 1), X_train];
X_test  = [ones(size(X_test, 1), 1), X_test];

lambda = 10;
I = eye(size(X_train, 2));
I(1,1) = 0;

% ridge
beta_ridge = (X_train' * X_train + lambda * I) \ (X_train' * y_train_std);
y_pred_std = X_test * beta_ridge;
y_pred = y_pred_std * y_std + y_mean;
mse_ridge = mean((y_test - y_pred).^2);
fprintf("Ridge MSE: %.4f\n", mse_ridge);

figure;
plot(y_test, y_pred, 'g.');
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--');
xlabel('Stvarne vrednosti');
ylabel('Predikcija');
title('Ridge regresija');
grid on;
legend('Predikcije', 'Idealno');

% lasso
lasso_f = @(b) sum((y_train_std - X_train * b).^2) + lambda * sum(abs(b(2:end)));
beta_init = zeros(size(X_train,2),1);
beta_lasso = fminunc(lasso_f, beta_init);

y_pred_std = X_test * beta_lasso;
y_pred = y_pred_std * y_std + y_mean;
mse_lasso = mean((y_test - y_pred).^2);
fprintf("Lasso MSE: %.4f\n", mse_lasso);

figure;
plot(y_test, y_pred, 'm.');
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--');
xlabel('Stvarne vrednosti');
ylabel('Predikcija');
title('Lasso regresija');
grid on;
legend('Predikcije', 'Idealno');

% KNN regresija
X_knn_raw = X_full_sel;
y_knn = y_full;

N = size(X_knn_raw, 1);
rand_idx = randperm(N);
X_knn_raw = X_knn_raw(rand_idx, :);
y_knn = y_knn(rand_idx);
n_train = round(0.7 * N);

X_train = X_knn_raw(1:n_train, :);
y_train = y_knn(1:n_train);
X_test = X_knn_raw(n_train+1:end, :);
y_test = y_knn(n_train+1:end);

mu_X = mean(X_train);
sigma_X = std(X_train);
X_train = (X_train - mu_X) ./ sigma_X;
X_test = (X_test - mu_X) ./ sigma_X;

y_mean = mean(y_train);
y_std = std(y_train);
y_train_std = (y_train - y_mean) / y_std;

K = 7;
N_test = size(X_test, 1);
y_pred_std = zeros(N_test, 1);

for i = 1:N_test
    x_i = X_test(i, :);
    dists = sum((X_train - x_i).^2, 2);
    [~, idx_sorted] = sort(dists);
    nearest_idx = idx_sorted(1:K);
    y_pred_std(i) = mean(y_train_std(nearest_idx));
end

y_pred = y_pred_std * y_std + y_mean;
mse_knn = mean((y_test - y_pred).^2);
fprintf("KNN regresija (K=%d) MSE: %.4f\n", K, mse_knn);

figure;
plot(y_test, y_pred, 'b.');
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--');
xlabel('Stvarne vrednosti');
ylabel('Predikcija');
title(sprintf('KNN regresija (K=%d)', K));
grid on;
legend('Predikcije', 'Idealno');
