import pandas as pd

def klasyfikuj_auto(auto):
    max_wynik = -1
    najlepsza_klasa = None

    for klasa in p_klasy.index:
        wynik = p_klasy[klasa]

        for cecha in auto.index:
            if cecha == 'class':
                continue
            wartosc_cechy = auto[cecha]

            p_cechy = p_cechy_dla_klasy[klasa][cecha].get(wartosc_cechy,1e-6)

            wynik = wynik*p_cechy

            if wynik > max_wynik:
                max_wynik = wynik
                najlepsza_klasa = klasa
    return najlepsza_klasa


nazwy_kolumn = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

df = pd.read_csv('car_evaluation.data', names=nazwy_kolumn)

df = df.sample(frac=1).reset_index(drop=True)

punkt_podzialu = int(len(df) * 0.8)

train_df = df[:punkt_podzialu]
test_df = df[punkt_podzialu:]
p_klasy = train_df['class'].value_counts(normalize=True)

p_cechy_dla_klasy = {}

for klasa in p_klasy.index:
    podzbior = train_df[train_df['class'] == klasa]

    p_cechy_dla_klasy[klasa] = {}

    for kolumna in train_df.columns[:-1]:
        prawdopodobienstwa = podzbior[kolumna].value_counts(normalize=True)

        p_cechy_dla_klasy[klasa][kolumna] = prawdopodobienstwa

dobre_odpowiedzi = 0


for i in range(len(test_df)):
    auto = test_df.iloc[i]
    
    prawdziwa_klasa = auto['class']
    
    przewidziana_klasa = klasyfikuj_auto(auto)

    if prawdziwa_klasa == przewidziana_klasa:
        dobre_odpowiedzi += 1
dokladnosc = dobre_odpowiedzi/len(test_df)*100
print(f"{dokladnosc}%")
