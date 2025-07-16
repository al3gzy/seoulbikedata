function [beta_konacno, izabrane_kolone, z_score] = backwards_stepwise(X_ulaz, y_izlaz)

X_trenutno = X_ulaz;
[N, D] = size(X_trenutno); % n - uzoraka, d - promenljivih

beta_full = (X_trenutno' * X_trenutno) \ (X_trenutno' * y_izlaz);
rss_stari = rss(X_trenutno, y_izlaz, beta_full);
sigma2 = rss_stari / (N - D);

varijansa_beta = diag(inv(X_trenutno' * X_trenutno) * sigma2);
z_score = beta_full ./ sqrt(varijansa_beta);

F_score = 0;
kriticna_vrednost = 1;
izabrane_kolone = 1:D;
izbacene_kolone = [];
izbaceni_indeksi = [];

while F_score < kriticna_vrednost
    [~, indeks_najslabije] = min(abs(z_score));

    % pamtimo koju kolonu izbacujemo
    poslednje_izbacena = izabrane_kolone(indeks_najslabije);
    izbacene_kolone = [izbacene_kolone, poslednje_izbacena];
    izbaceni_indeksi = [izbaceni_indeksi, indeks_najslabije];

    % uklanjamo tu kolonu iz podataka i indeksa
    X_trenutno(:, indeks_najslabije) = [];
    izabrane_kolone(indeks_najslabije) = [];

    % izracunavamo novi model
    beta_novi = (X_trenutno' * X_trenutno) \ (X_trenutno' * y_izlaz);
    rss_novi = rss(X_trenutno, y_izlaz, beta_novi);

    % nova procena var i z, racunamo F
    sigma2 = rss_novi / (N - size(X_trenutno, 2));
    varijansa_beta = diag(inv(X_trenutno' * X_trenutno) * sigma2);
    z_score = beta_novi ./ sqrt(varijansa_beta);
    broj_izbacenih = length(izbaceni_indeksi);
    F_score = ((rss_novi - rss_stari) / broj_izbacenih) / (rss_stari / (N - D));

    % kriticna vrednost
    kriticna_vrednost = chi2inv(0.95, broj_izbacenih) / broj_izbacenih;
end

% vracamo poslednju kolonu
izabrane_kolone = [izabrane_kolone, poslednje_izbacena];

% finalni model
X_konacno = X_ulaz(:, izabrane_kolone);
beta_konacno = (X_konacno' * X_konacno) \ (X_konacno' * y_izlaz);
sigma2 = sum((y_izlaz - X_konacno * beta_konacno).^2) / (size(X_konacno, 1) - size(X_konacno, 2));
varijansa_beta = diag(inv(X_konacno' * X_konacno) * sigma2);
z_score = beta_konacno ./ sqrt(varijansa_beta);

disp("Optimalne kolone dobijene backward stepwise selekcijom su:");
disp(izabrane_kolone);

end
