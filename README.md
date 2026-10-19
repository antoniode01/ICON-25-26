# ICON — Industrial Diagnosis

Sistema dimostrativo di diagnosi dei guasti industriali che integra **ontologia OWL**, **inferenza semantica** e **apprendimento automatico**.

Il progetto confronta una rappresentazione di base dei casi con una rappresentazione arricchita da conoscenza di dominio (*Ontology-Based Knowledge*, OntoBK). Il dataset incluso è **sintetico**: i risultati descrivono lo scenario simulato e non costituiscono una validazione su impianti industriali reali.

## Funzionalità

- Ontologia OWL per macchine, componenti, sensori, osservazioni, sintomi e famiglie di guasto.
- Classificazione semantica dei sintomi mediante assiomi OWL e reasoner HermiT.
- Diagnosi simbolica con compatibilità macchina-guasto, evidenze favorevoli e sintomi associati dichiarati assenti.
- Confronto fra rappresentazioni **Base** e **OntoBK** con Random Forest, SVM RBF e MLP.
- Cross-validation stratificata ripetuta (5 fold × 3 ripetizioni), metriche macro e matrici di confusione.
- Test automatici con pytest.

## Requisiti

- Python 3.10 o successivo consigliato.
- Java installato e disponibile nel sistema per l'esecuzione di HermiT tramite Owlready2.
- Dipendenze Python elencate in `requirements.txt`.

## Installazione (Windows / PowerShell)

Aprire PowerShell nella cartella principale del progetto:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se PowerShell impedisce l'attivazione dell'ambiente virtuale, si può usare direttamente `.venv\Scripts\python.exe`.

## Avvio

Dalla cartella principale:

```powershell
python -m src.main
```

All'avvio il programma costruisce l'ontologia, avvia HermiT e salva l'ontologia in `results/industrial_diagnosis.owl`. Si apre quindi il menu:

1. **Diagnosi simbolica** — selezione della macchina e inserimento dello stato dei sintomi.
2. **Valutazione Machine Learning** — esecuzione del confronto Base/OntoBK e aggiornamento dei CSV in `results/`.
3. **Riepilogo dell'ontologia** — statistiche della base di conoscenza.
4. **Esci**.

Per ogni sintomo si può scegliere presente, assente, sconosciuto o non applicabile. Premere Invio senza scelta equivale a sconosciuto.

## Dataset

`data/industrial_faults.csv` contiene 3.000 casi e 22 colonne: identificativo `case_id`, attributi descrittivi e target `fault`. Le quattro classi sono `GUASTO_TERMICO`, `GUASTO_MECCANICO`, `GUASTO_ELETTRICO` e `GUASTO_IDRAULICO`, con 750 casi ciascuna nel file incluso.

Sono rappresentati quattro tipi di macchina: motore elettrico, pompa industriale, compressore e nastro trasportatore. I guasti idraulici sono assegnati solo a pompe e compressori; le misure e i sintomi di pressione/portata risultano non applicabili per motori elettrici e nastri trasportatori.

Il generatore è `src/dataset_generator.py`. Per rigenerare il dataset:

```powershell
python -m src.dataset_generator
```

Questo comando sovrascrive il CSV usando il seed predefinito `42`. Per riprodurre esattamente gli esperimenti inclusi, non rigenerare il dataset prima di confrontare i risultati salvati.

## Valutazione sperimentale

La valutazione usa `RepeatedStratifiedKFold` con 5 fold e 3 ripetizioni, seed `42`. L'estrattore semantico è adattato all'interno di ciascun fold di training; imputazione e standardizzazione sono incluse nella pipeline e apprese sul solo training fold. Sono riportate accuracy, precision, recall e F1 macro, con deviazione standard fra i 15 fold.

Risultati registrati in `results/comparison_results.csv`:

| Modello | Rappresentazione | Accuracy media | F1 macro medio |
|---|---|---:|---:|
| Random Forest | Base | 0,9032 | 0,9030 |
| Random Forest | OntoBK | 0,8752 | 0,8747 |
| SVM RBF | Base | 0,9084 | 0,9082 |
| SVM RBF | OntoBK | 0,8981 | 0,8978 |
| MLP | Base | 0,9139 | 0,9140 |
| MLP | OntoBK | 0,9146 | 0,9144 |

L'arricchimento OntoBK non migliora automaticamente tutti i modelli: nei risultati inclusi le prestazioni sono inferiori per Random Forest e SVM, mentre per MLP la differenza è minima. Il risultato va interpretato e discusso, non nascosto: rappresentare più conoscenza non garantisce da solo una maggiore accuratezza predittiva.

Le matrici di confusione aggregano le predizioni out-of-fold delle tre ripetizioni: ogni matrice contiene complessivamente 9.000 predizioni (3.000 casi × 3 ripetizioni).

## Test

Dalla cartella principale:

```powershell
python -m pytest -q
```

I test coprono generazione e proprietà del dataset, struttura dell'ontologia, reasoner, estrazione delle feature e costruzione delle pipeline. Per eseguire la suite completa sono necessarie tutte le dipendenze di `requirements.txt` e un ambiente in cui HermiT sia avviabile.

## Struttura del progetto

```text
ICON_Industrial_Diagnosis/
├── data/
│   └── industrial_faults.csv
├── docs/
│   └── Relazione_tecnica_ICON_Industrial_Diagnosis.pdf
├── results/
│   ├── industrial_diagnosis.owl
│   ├── comparison_results.csv
│   └── confusion_*.csv
├── src/
│   ├── dataset_generator.py
│   ├── main.py
│   ├── ontology.py
│   ├── semantic_reasoner.py
│   └── ontoBK_learning.py
├── tests/
├── requirements.txt
└── README.md
```

## Limiti e interpretazione

1. Il dataset è sintetico e deriva da ipotesi probabilistiche e regole del generatore; non sostituisce dati raccolti su macchinari reali.
2. L'ontologia codifica una conoscenza di dominio semplificata a scopo didattico.
3. Il supporto della diagnosi simbolica è un conteggio di evidenze favorevoli, **non una probabilità** né una stima calibrata del rischio.
4. Le prestazioni dipendono dal generatore, dalle feature e dai modelli selezionati e non vanno generalizzate direttamente a contesti industriali reali.
5. L'arricchimento semantico può migliorare la rappresentazione esplicita e la spiegabilità anche quando non aumenta l'accuratezza.

## Obiettivo didattico

Il progetto integra rappresentazione della conoscenza, inferenza semantica e apprendimento supervisionato. Il confronto sperimentale tra due rappresentazioni degli stessi casi consente di discutere in modo critico vantaggi e limiti dell'integrazione OntoBK.
