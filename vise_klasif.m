pkg load io
pkg load statistics

raw = csv2cell('SeoulBikeData.csv');
headers = raw(1, :);
data = raw(2:end, :);
idx_rent = find(strcmp(headers, 'Rented Bike Count'));
idx_season = find(strcmp(headers, 'Seasons'));
idx_holiday = find(strcmp(headers, 'Holiday'));
idx_funcday = find(strcmp(headers, 'Functioning Day'));
idx_features = [4 5 6 7 8 9 10 11];

function r = encode(data, non_numeric_columns)
  r = [];
  for col = non_numeric_columns
    unique_values = unique(data(:, col));
    encoded_col = zeros(size(data,1), length(unique_values)-1);

    for i = 1:length(unique_values)
      encoded_vector = strcmp(data(:, col), unique_values{i});

      if i < length(unique_values)
        encoded_col(:, i) = encoded_vector;
      end
    end

    r = [r, encoded_col];
  end
end

X_num = cell2mat(data(:, idx_features));
X_cat = encode(data, [idx_season idx_holiday idx_funcday]);
X = [X_num, X_cat];

% y sa 5 klasa na osnovu kvantila
y_raw = cell2mat(data(:, idx_rent));
edges = quantile(y_raw, [0.2 0.4 0.6 0.8]);
y = zeros(size(y_raw));
y(y_raw <= edges(1)) = 1;
y(y_raw > edges(1) & y_raw <= edges(2)) = 2;
y(y_raw > edges(2) & y_raw <= edges(3)) = 3;
y(y_raw > edges(3) & y_raw <= edges(4)) = 4;
y(y_raw > edges(4)) = 5;

% K-Fold
N = size(X,1);
k = 5;
part = cvpartition(N, 'KFold', k);
part = struct(part);

% KNN funkcija
function ypred = knn_klasifikator(xts, xtr, ytr, k)
  nts = size(xts,1);
  ypred = zeros(nts,1);
  for i = 1:nts
    dist = sqrt(sum((xtr - xts(i,:)).^2, 2));
    [~, idx] = sort(dist);
    najblizi_idx = idx(1:k);
    susedi_lab = ytr(najblizi_idx);
    ypred(i) = mode(susedi_lab);
  end
end

% K-Fold evluacija
err_knn = zeros(1,k);
err_lin = zeros(1,k);
err_lda = zeros(1,k);
err_qda = zeros(1,k);

for fold = 1:k
  idtr = (part.inds != fold);
  idval = (part.inds == fold);

  xtr = X(idtr, :);
  ytr = y(idtr);
  xts = X(idval, :);
  yts = y(idval);

  % standardizacija
  [xtr, mu, sigma] = zscore(xtr);
  xts = (xts - mu) ./ sigma;

  xtr_aug = [ones(size(xtr,1),1) xtr];
  xts_aug = [ones(size(xts,1),1) xts];

  % KNN
  ypred_knn = knn_klasifikator(xts, xtr, ytr, 5);
  err_knn(fold) = mean(ypred_knn != yts);

  % Linearna klasifikacija
  classes = unique(ytr);
  g = length(classes);
  y_encoded = zeros(length(ytr), g);
  for i=1:g
    y_encoded(:,i) = (ytr == classes(i));
  end

  beta = (xtr_aug' * xtr_aug) \ (xtr_aug' * y_encoded);
  y_pred_lin = xts_aug * beta;
  [~, ypred_lin_classes] = max(y_pred_lin, [], 2);
  ypred_lin_map = classes(ypred_lin_classes);

  err_lin(fold) = mean(ypred_lin_map != yts);

  % LDA i QDA

  pi_full = zeros(g,1);
  mi = zeros(size(xtr,2), g);
  sig_p = zeros(size(xtr,2));
  delta_qda = zeros(length(yts), g);

  for i=1:g
    x_klasa = xtr(ytr==classes(i), :);
    pi_k = size(x_klasa,1) / size(xtr,1);
    mi_k = mean(x_klasa, 1)';
    mi(:, i) = mi_k;
    pi_full(i) = pi_k;

    x_cent = x_klasa - repmat(mi_k', size(x_klasa,1), 1);

    % Za LDA
    for r=1:size(x_klasa,1)
      sig_p = sig_p + (x_cent(r,:)' * x_cent(r,:));
    end

    % Za QDA
    sig_k = zeros(size(xtr,2));
    for r=1:size(x_klasa,1)
      sig_k = sig_k + (x_cent(r,:)' * x_cent(r,:));
    end
    sig_k = sig_k / (size(x_klasa,1) - 1);
    % epsilon = 1e-1;
    % sig_k = sig_k + epsilon * eye(size(sig_k)); regularizacija matrice
    inv_sig_k = inv(sig_k);
    diff = xts - repmat(mi_k', size(xts,1), 1);
    delta_qda(:, i) = -0.5*log(det(sig_k)) - 0.5*sum((diff * inv_sig_k).*diff, 2) + log(pi_k);
  end

  sig_p = sig_p / (size(xtr,1) - g);
  inv_sig_p = inv(sig_p);
  delta_lda = zeros(length(yts), g);

  for i=1:g
    delta_lda(:, i) = xts * inv_sig_p * mi(:,i) - 0.5 * (mi(:,i)' * inv_sig_p * mi(:,i)) + log(pi_full(i));
  end

  [~, ypred_lda_idx] = max(delta_lda, [], 2);
  ypred_lda = classes(ypred_lda_idx);
  err_lda(fold) = mean(ypred_lda != yts);

  [~, ypred_qda_idx] = max(delta_qda, [], 2);
  ypred_qda = classes(ypred_qda_idx);
  err_qda(fold) = mean(ypred_qda != yts);

end

fprintf('Prosečna greska KNN: %.4f\n', mean(err_knn));
fprintf('Prosečna greska Linearne klasifikacije: %.4f\n', mean(err_lin));
fprintf('Prosečna greska LDA: %.4f\n', mean(err_lda));
fprintf('Prosečna greska QDA: %.4f\n', mean(err_qda));

acc_knn = 1 - err_knn;
acc_lin = 1 - err_lin;
acc_lda = 1 - err_lda;
acc_qda = 1 - err_qda;

folds = 1:k;
figure;
plot(folds, acc_knn, '-o', 'DisplayName', 'KNN');
hold on;
plot(folds, acc_lin, '-s', 'DisplayName', 'Linearna klasifikacija');
plot(folds, acc_lda, '-d', 'DisplayName', 'LDA');
plot(folds, acc_qda, '-^', 'DisplayName', 'QDA');
xlabel('Fold');
ylabel('Preciznost (Accuracy)');
legend('KNN', 'Linearna klasifikacija', 'LDA', 'QDA');grid on;
hold off;
