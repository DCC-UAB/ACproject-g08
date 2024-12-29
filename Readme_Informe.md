# **Informe del projecte (Grup 08): Sistemes Recomanadors de Pel·lícules**

## Índex
1. **Introducció**
2. **Instal·lació i Configuració**
   - Requisits
   - Descarrega de la Base de Dades
   - Inicialització de les Datasets
3. **Arquitectura del Projecte**
   - Estructura del Codi
   - Estructura d'Arxius
   - Descripció d'Arxius i Funcionalitats
4. **Models de Recomanació**
   - Basat en Contingut
   - Col·laboratiu (Usuari-Usuari)
   - Descomposició en Valors Singulars (SVD)
5. **Guia d'ús**
   - Execució Principal
   - Execució Individual de Models
6. **Anàlisi i Resultats**
   - Visualització de Resultats
   - Comparació de Models
7. **Preguntes Obertes i Millores Potencials**
8. **Crèdits i Llicència**

---

## 1. Introducció
Aquest projecte té com a objectiu explorar i avaluar diferents enfocaments per a sistemes de recomanació de pel·lícules utilitzant informació d'interaccions prèvies dels usuaris i característiques de les pel·lícules. S'utilitzen datasets com `ratings_small.csv` i `movies_metadata.csv`.

## 2. Instal·lació i Configuració

### Requisits
- Python 3.8+
- Llibreries: `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`

### Descarrega de la Base de Dades
1. Descarrega els datasets des de [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).
2. Col·loca els arxius a la carpeta `Data/`.

### Inicialització
Executa:
```bash
python ./Data_Acces/cleaner.py
```
Això netejarà les dades per preparar-les.

---

## 3. Arquitectura del Projecte

### Estructura del Codi
Organització modular per facilitar l'extensibilitat i el manteniment.

### Estructura d'Arxius
```plaintext
├── Data_Acces/
│   ├── cleaner.py                      # Neteja inicial de dades.
├── recomanadors/
│   ├── funcio_principal.py             # Punt d’entrada principal.
│   ├── svd.py                          # Implementació del model SVD.
│   ├── user_model.py                   # Model col·laboratiu user-to-user.
│   ├── extreure_metadata.py            # Visualització de metadades.
│   ├── content_based.py                # Model basat en contingut.
│   ├── user_to_user_experiments.py     # Experiments addicionals.
├── Data/
│   ├── movies_metadata.csv             # Metadades de pel·lícules.
│   ├── ratings_small.csv               # Valoracions dels usuaris.
```

### Descripció d'Arxius
- `user_model.py`: Càlcul de similituds entre usuaris i predicció de valoracions.
- `content_based.py`: Recomanacions basades en paraules clau i gèneres de pel·lícules.
- `svd.py`: Implementació de l'algorisme SVD per a recomanacions.
- `extreure_metadata.py`: Extracció d'informació detallada sobre pel·lícules.

---

## 4. Models de Recomanació

### Basat en Contingut
Aquest model utilitza les paraules clau i els gèneres associats a les pel·lícules per trobar similituds entre les que ja han estat valorades per l'usuari i les que no. Calcula aquestes similituds mitjançant el coeficient de Dice per determinar quines podrien ser del seu interès. És especialment efectiu per a usuaris que no han valorat moltes pel·lícules, ja que explota les característiques intrínseques de les pel·lícules.

### Col·laboratiu Usuari-Usuari
Aquest mètode compara les preferències d'un usuari amb les d'altres per trobar patrons comuns. Utilitza una matriu de valoracions on les files representen usuaris i les columnes representen pel·lícules. Basat en la similitud de cosinus entre usuaris, aquest model recomana pel·lícules que han estat ben valorades per altres usuaris amb gustos similars. És ideal per a usuaris actius amb moltes valoracions prèvies.

### SVD
La descomposició en valors singulars (SVD) és un mètode avançat que descompon la matriu d'usuari-pel·lícula en tres components per identificar patrons latents (en el nostre cas, les preferències dels usuaris) en les dades. Això permet predir valoracions mancants amb alta precisió. Aquest model és eficaç per a sistemes amb grans quantitats de dades i és capaç de gestionar problemes d'esparsitat a la matriu de valoracions.

---

## 5. Guia d'ús

### Execució Principal
Executa:
```bash
python funcio_principal.py
```
Selecciona el model que vulguis utilitzar.

### Execució Individual
Per provar models individualment:
```bash
python user_model.py
python content_based.py
python svd.py
```

---

## 6. Anàlisi i Resultats
Els resultats inclouen mètriques com RMSE, MAE, i gràfics comparatius entre prediccions i valors reals. Així es possible identificar fortaleses i debilitats en diferents escenaris.

---

## 7. Preguntes Obertes i Millores Potencials
- És possible millorar la precisió utilitzant un híbrid entre models?
- Quines millores pot aportar un preprocesament més robust de les dades?

---

## 8. Crèdits i Llicència
Desenvolupat pel Grup 08 com a part del projecte de final d'assignatura d'Aprenentatge Computacional.
