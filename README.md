[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/USx538Ll)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=17349155&assignment_repo_type=AssignmentRepo)

# <center> Projecte Grup 08: Sistema Recomanador de Pel·lícules </center>

Aquest sistema recomanador està dissenyat per explorar i avaluar diferents enfocaments per recomanar pel·lícules als usuaris en funció de les seves interaccions prèvies i les característiques de les pel·lícules. Les dades provenen d'un conjunt reduït de qualificacions (ratings_small.csv) i metadades de pel·lícules (movies_metadata.csv)

## Instruccions d'execució
#### Descarrega de la BD i inicialització de les dades
1. Descarregar el la BD del següent link: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv
2. Executar l'arxiu cleaner.py de la carpeta Data_Acces 
#### Executar Recomanadors
- Dins de la carpeta recomanadors està l'arxiu funcio_principal.py des d'on es poden executar tots els recomanadors
- Si ja has inicialitzat les dades, pots entrar a cada recomanador de manera separada, cadascún té un __main__ per fer recomanacions y probes amb el propi recomanador
- Tots els arxius tenen una funció per printejar les dades de manera clara, a part del id de la DB
## Seguiment
### Setmana 1
#### Objectius primera setmana:
- Leer base de datos
- Poner base de datos en formato database (pandas)
- Visualizar datos (peliculas mas votadas, con rating global más alto, etc...)
- Limpiar database <- No prioritario
### Setmana 2
#### Per a la següent sessió de seguiment, volem complir els següents objectius:
- Implementació del models col·laboratius item-item i user-based
- Implementació del model basat en contingut
- Neteja de la base de dades

#### També volem ser capaços de respondre a les següents preguntes:
- Surt a compte netejar la base de dades? Millora el rendiment dels recomanadors?
    - Efectivament, la netja de la base de dades, tot i que encara simple, pot reduïr en gran quantitat el temps d'execució dels models
- Quin model de recomendació és millor pels usuaris que han puntuat més pel·lícules? I pels que han puntuat menys?
- Algún model afavoreix a que s'escolleixin pel·lícules amb més puntuacions? I amb menys?

### Setmana 3
#### Per a la següent sessió de seguiment, volem complir els següents objectius:
- Implementar SVD
- Comprovar que els models funcionan correctament
- Graficar les dades

### Preguntes Finals
#### Amb el projecte finalitzat, encara hi han més preguntes a respondre:
- Es pot millorar l'accuracy dels recomanadors?
- Existeixen realment clusters diferenciats pels que separar la BD?
- Una implemnetació creuada entre models podria millorar les recomanacions o provoca overfitting?