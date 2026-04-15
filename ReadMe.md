# MLOps_sentiment_project per CI/CD e Deploy

## STRUTTURA DEL PROGETTO

- **data/**  
  Contiene il dataset utilizzato per il training → *Twitter_Data.csv*

- **scripts/**  
  Contiene tutti gli script principali del progetto:  
  - preprocess.py  
  - train.py  
  - evaluate_and_save.py  
  - deploy_huggingface.py  
  - monitor.py

- **.github/workflows/**  
  Contiene la pipeline CI/CD (*mlops_pipeline.yml*)

- **notebooks/**  
  Notebook Google Colab per eseguire e testare l’intero progetto

---

## DESCRIZIONE DEGLI SCRIPT

### preprocess.py  
Carica e prepara i dati:  
- Legge un CSV da URL o percorso locale  
- Controlla che esistano le colonne “clean_text” e “category”  
- Rimuove i valori nulli  
- Converte i tipi di dato  
- Applica shift alle etichette da (-1,0,1) → (0,1,2) per facilitare il training  
- Divide il dataset in train e test (split stratificato)  
Output: X_train, X_test, y_train, y_test

### train.py  
Fine-tuning del modello con partial freezing del backbone RoBERTa:  
- Usa il modello HuggingFace *cardiffnlp/twitter-roberta-base-sentiment-latest*  
- Converte i dati in formato HuggingFace Dataset  
- Tokenizza i testi  
- Allena il modello con Trainer e valuta durante il training  
- Salva il modello fine-tunato  
Output: modello salvato in `./data/results/trained_model`

### evaluate_and_save.py  
Script MLOps per valutazione e deploy controllato:  
- Valuta il modello appena addestrato calcolando F1-score macro  
- Scarica il modello precedente da HuggingFace Hub (se esiste)  
- Confronta le performance:  
  - Se il nuovo modello è migliore o uguale → lo pubblica sul Hub  
  - Altrimenti non fa nulla  
- Garantisce un deploy automatico controllato e sicuro

### deploy_huggingface.py  
Script per eseguire inferenza usando il modello fine-tunato caricato da HuggingFace Hub:  
- Carica modello e tokenizer dal Hub  
- Crea pipeline di sentiment analysis  
- Restituisce predizioni con sentiment tradotto in etichette leggibili

### monitor.py  
Sistema di monitoraggio continuo delle performance del modello:  
- Campiona casualmente un sottoinsieme del dataset di test a intervalli regolari (es. ad ogni run della pipeline)  
- Valuta F1 macro del modello sul campione  
- Registra metriche e distribuzione predizioni in un log JSON versionato nel repo  
- Imposta una soglia di alert (default F1=0.7) per segnalare degrado di performance  
- Solleva eccezione se il modello scende sotto la soglia, suggerendo la necessità di retraining  
- Questo sistema permette di tenere sotto controllo la qualità del modello in produzione e intervenire tempestivamente

---

## PIPELINE CI/CD (GitHub Actions)

**File:** `mlops_pipeline.yml`  

La pipeline si attiva automaticamente quando:  
- Viene fatto push sul branch principale (`main`)  
- Viene eseguito manualmente tramite workflow dispatch  

**Step principali:**  
- Clona il repository  
- Installa le dipendenze tramite `pip`  
- Esegue il training del modello  
- Esegue la valutazione e confronta il modello con quello precedente  
- Effettua il deploy se il modello è migliorato  
- Esegue il monitoraggio delle performance e salva i log  
- Versiona il log di monitoraggio sul repo per storico

---

## NOTE SUL TRAINING E RISORSE

Per limitazioni di risorse e tempistiche (uso di ambienti gratuiti come Colab), il training e la valutazione usano sottoinsiemi ridotti del dataset. Aumentando i dati e potendo utilizzare hardware più potente si possono ottenere metriche migliori (accuracy e F1 superiori a 0.9).

---

## MODELLO UTILIZZATO

`cardiffnlp/twitter-roberta-base-sentiment-latest`  
Modello transformer pre-addestrato per analisi del sentiment su Twitter, fine-tunato sul dataset fornito.

---

## DATASET

**Twitter sentiment dataset**

- Colonne:  
  - `clean_text`: testo  
  - `category`: etichetta sentiment originale (-1=negativo, 0=neutro, 1=positivo), shiftata a (0,1,2) nel preprocessing

- Classi:  
  - 0 = negativo  
  - 1 = neutro  
  - 2 = positivo

---

```shiftata a (0,1,2) nel preprocessing
Classi:

0 = negativo
1 = neutro
2 = positivo
