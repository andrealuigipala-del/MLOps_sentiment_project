MLOps_sentiment_project for CI/CD deploy

STRUTTURA DEL PROGETTO

data/
Contiene il dataset utilizzato per il training
-> Twitter_Data.csv

scripts/
Contiene tutti gli script principali del progetto:

- preprocess.py
- train.py
- evaluate_and_save.py
- deploy_huggingface.py
- deploy_local.py

.github/workflows/
Contiene la pipeline CI/CD (mlops_pipeline.yml)
notebook_colab.ipynb
Notebook per eseguire e testare tutto il progetto

----------------------------------------------------------------

NB: per questioni di ottimizzazine delle risorse (e delle tempistiche), poiché sono state usate quelle gratuite (limitate) si è proceduto ad usare il dataset parzialmente, sia in fase di training, sia in fase di valutazione. Aumentando la quantità di dati (e avendo tempo a disposizione) è possibile avere metriche promettenti (accuracy f1 > 0.9)

----------------------------------------------------------------

DESCRIZIONE DEGLI SCRIPT

--- preprocess.py
Questo script serve per caricare e preparare i dati.

Legge un CSV da URL o percorso locale
Controlla che esistano le colonne “clean_text” e “category”
Rimuove i valori nulli
Converte i tipi di dato
Divide il dataset in train e test (split stratificato)

Output:
X_train, X_test, y_train, y_test

--- train.py
Questo script esegue il fine-tuning del modello.

Usa il modello HuggingFace:
cardiffnlp/twitter-roberta-base-sentiment-latest
Converte i dati in formato HuggingFace Dataset
Tokenizza i testi
Addestra il modello con Trainer
Valuta durante il training
Salva il modello finale

Output:
Il modello viene salvato in:
./data/results/final_model

--- evaluate_and_save.py
Questo è lo script principale per la logica MLOps.

Funziona così:

Valuta il modello appena addestrato (F1-score)
Scarica il modello precedente da Hugging Face (se esiste)
Confronta le performance
Se il nuovo modello è migliore o uguale → lo pubblica
Se è peggiore → non fa nulla

Questo permette un deploy automatico controllato.

--- deploy_huggingface.py
Serve per usare il modello direttamente da Hugging Face.

Carica modello e tokenizer dal Hub
Crea una pipeline di sentiment analysis
Restituisce predizioni su nuovi testi

--- deploy_local.py
Serve per usare il modello salvato in locale.

Carica il modello dalla cartella locale
Esegue predizioni su nuovi testi

------------------------------------

PIPELINE CI/CD (GitHub Actions)

--- File: mlops_pipeline.yml

La pipeline si attiva quando:

viene fatto push su main
oppure manualmente

Step:

- Clona il repository
- Installa le dipendenze
- Esegue il training
- Salva il modello come artifact
- Ricarica il modello
- Esegue la valutazione
- Confronta con modello precedente
- Eventuale deploy su Hugging Face

-------------------------------------

NOTEBOOK GOOGLE COLAB

Il notebook serve per testare manualmente il progetto.

Cosa fa:

- Installa librerie
- Clona il repository
- Importa gli script
- Carica il dataset
- Allena il modello (subset dei dati)
- Valuta le performance (F1-score)
- Testa le predizioni
- Permette anche push su GitHub

MODELLO UTILIZZATO

cardiffnlp/twitter-roberta-base-sentiment-latest

Modello transformer pre-addestrato per sentiment analysis.
Viene fine-tuned sul dataset Twitter.

------------------------------------------

DATASET

- Twitter sentiment dataset

Colonne:

- clean_text → testo
- category → etichetta

Classi:
- 0 = negativo
- 1 = neutro
- 2 = positivo
