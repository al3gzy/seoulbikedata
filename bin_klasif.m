pkg load io
pkg load statistics

raw = csv2cell('SeoulBikeData.csv');
headers = raw(1, :);
data = raw(2:end, :);

idx_rent = find(strcmp(headers, 'Rented Bike Count'));
idx_season = find(strcmp(headers, 'Seasons'));
idx_holiday = find(strcmp(headers, 'Holiday'));
idx_funcday = find(strcmp(headers, 'Functioning Day'));
idx_features = [3 4 5 6 7 8 9 10 11];

function r = encode(data, non_numeric_columns) % sa vežbi
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

% umesto csvread koristimo cell2mat jer ne radi lepo na X_cat
X_num = cell2mat(data(:, idx_features));
X_cat = encode(data, [idx_season, idx_holiday, idx_funcday]);
X = [X_num, X_cat];

y_raw = cell2mat(data(:, idx_rent));
y = double(y_raw >= median(y_raw)); # klase 0 i 1

% K-Fold
pkg load statistics
N = size(X,1);
k = 5;
part = cvpartition(N, 'KFold', k);
part = struct(part);
err_lin = [];

for i = 1:k
  idtr = (part.inds != i);
  idts = (part.inds == i);

  xtr = X(idtr, :); ytr = y(idtr);
  xts = X(idts, :); yval = y(idts);

  % standardizacija
  [xtr, mu, sigma] = zscore(xtr);
  xts = (xts - mu) ./ sigma;

  % kolona jedinica
  xtr = [ones(size(xtr,1),1), xtr];
  xts = [ones(size(xts,1),1), xts];

  % linearna regresija
  beta = (xtr' * xtr) \ (xtr' * ytr);
  ypred = xts * beta;

  % binarizacija
  ypred = ypred >= 0.5;
  err = mean(ypred != yval);
  err_lin = [err_lin, err];
end

fprintf('Prosečna greška K-Fold Lin: %.4f\n', mean(err_lin));

% KNN klasifikacija
function ypred = knn_klasifikator(xts, xtr, ytr, k)
  nval = size(xts,1);
  ypred = zeros(nval,1);

  for i = 1:nval
    dist = sqrt(sum((xtr - xts(i,:)).^2, 2));
    [~, idx] = sort(dist);
    najblizi_idx = idx(1:k);
    susedi_lab = ytr(najblizi_idx);
    ypred(i) = mode(susedi_lab);
  end
end

err_knn = [];

for i = 1:k
  idtr = (part.inds != i);
  idts = (part.inds == i);

  xtr = X(idtr, :); ytr = y(idtr);
  xts = X(idts, :); yval = y(idts);

  % standardizacija
  [xtr, mu, sigma] = zscore(xtr);
  xts = (xts - mu) ./ sigma;

  ypred = knn_klasifikator(xts, xtr, ytr, 5);
  err = mean(ypred != yval);
  err_knn = [err_knn, err];
  if i == k
    TP = sum((ypred == 1) & (yval == 1));
    TN = sum((ypred == 0) & (yval == 0));
    FP = sum((ypred == 1) & (yval == 0));
    FN = sum((ypred == 0) & (yval == 1));

    fprintf('Konfuziona matrica KNN za fold %d (TP, FP, FN, TN): %d %d %d %d\n', i, TP, FP, FN, TN);
    fprintf('         Pred=0  Pred=1\n');
    fprintf('Stvar=0:   %4d    %4d\n', TN, FP);
    fprintf('Stvar=1:   %4d    %4d\n', FN, TP);
  end
end

fprintf('Prosečna greška KNN: %.4f\n', mean(err_knn));

% Logistička regresija
err_log = [];

sigmoid = @(z) 1 ./ (1 + exp(-z));
costFunction = @(theta, X, y) ...
   (-y .* log(sigmoid(X * theta)) - (1 - y) .* log(1 - sigmoid(X * theta)));

for i = 1:k
  idtr = (part.inds != i);
  idts = (part.inds == i);

  xtr = X(idtr, :); ytr = y(idtr);
  xts = X(idts, :); yval = y(idts);

  % standardizacija
  [xtr, mu, sigma] = zscore(xtr);
  xts = (xts - mu) ./ sigma;

  % kolona jedinica
  xtr = [ones(size(xtr,1),1), xtr];
  xts = [ones(size(xts,1),1), xts];

  [m, n] = size(xtr);
  initial_theta = zeros(n,1);
  [theta, ~] = fminunc(@(t)(mean(costFunction(t, xtr, ytr))), initial_theta);

  % predikcija
  ypred_prob = sigmoid(xts * theta);
  ypred_bin = ypred_prob >= 0.5;

  err = mean(ypred_bin != yval);
  err_log = [err_log, err];

  if i == k
    % ispis mat konfuzije za poslednji fold
    TP = sum((ypred_bin == 1) & (yval == 1));
    TN = sum((ypred_bin == 0) & (yval == 0));
    FP = sum((ypred_bin == 1) & (yval == 0));
    FN = sum((ypred_bin == 0) & (yval == 1));

    fprintf('Konfuziona matrica za fold %d (TP, FP, FN, TN): %d %d %d %d\n', i, TP, FP, FN, TN);
    fprintf('         Pred=0  Pred=1\n');
    fprintf('Stvar=0:   %4d    %4d\n', TN, FP);
    fprintf('Stvar=1:   %4d    %4d\n', FN, TP);
  end
end

fprintf('Prosečna greška Logističke regresije: %.4f\n', mean(err_log));
