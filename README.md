# ICON Course Project 2024/2025

Questo repository raccoglie il progetto sviluppato per il corso di ICON
2024/2025, tenuto dal Professor Nicola Fanizzi presso l'Università degli
Studi di Bari Aldo Moro.

L'obiettivo del lavoro è mettere in pratica le conoscenze acquisite durante il
corso, applicandole all'analisi di un dataset estratto dal videogioco
competitivo online **League of Legends (LoL)**.

[League of Legends](https://it.wikipedia.org/wiki/League_of_Legends) è un MOBA
(Multiplayer Online Battle Arena) in cui due squadre da cinque giocatori si
affrontano con lo scopo di distruggere la base avversaria. Ogni giocatore
controlla un "campione", dotato di abilità uniche e ruoli differenti (es.
supporto, tiratore, mago, tank). La partita si sviluppa in diverse fasi (early,
mid e late game), caratterizzate da strategie, combattimenti e gestione delle
risorse.

[![Apri la documentazione](https://img.shields.io/badge/Documentazione-PDF-red?style=for-the-badge&logo=adobeacrobatreader)](docs/Documentazione.pdf)

## Installazione

Clonare il repository:

```bash
git clone https://github.com/DomenicoBianchini/ICON.git
```

Installare le dipendenze:

```bash
pip install -r requirements.txt
```

## Struttura del progetto

Le cartelle principali del progetto sono:

- `datasets`: contiene i dataset utilizzati per l'analisi;
- `pre_processing`: contiene gli script per la preparazione dei dati;
- `supervised_training`: contiene gli script per l'apprendimento supervisionato;
- `bayes`: contiene il classificatore bayesiano;
- `prolog`: contiene la knowledge base e le relative interrogazioni;
- `displayers`: contiene gli script per la visualizzazione dei risultati;
- `images`: contiene i grafici generati;
- `docs`: contiene la documentazione del progetto.

Il file principale è `main.py`.

## Esecuzione

Il progetto può essere avviato con:

```bash
python main.py
```

## Documentazione

La documentazione completa del progetto è disponibile in formato PDF:

[**Apri la documentazione**](docs/Documentazione.pdf)

È disponibile anche la versione in formato DOCX:

[**Apri la documentazione in formato DOCX**](docs/Documentazione.docx)

## Autore

Progetto sviluppato da **Domenico Bianchini**, Università degli Studi di Bari
Aldo Moro.
