
"""Punto di ingresso del sistema ICON — Industrial Diagnosis."""

from pathlib import Path

import pandas as pd

from src.ontology import build_ontology, ontology_summary
from src.semantic_reasoner import DiagnosticReasoner
from src.ontoBK_learning import compare_representations


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "industrial_faults.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

MACHINES = {
    "1": ("Motore elettrico", "electric_motor"),
    "2": ("Pompa industriale", "industrial_pump"),
    "3": ("Compressore", "compressor"),
    "4": ("Nastro trasportatore", "conveyor_belt"),
}

SYMPTOMS = {
    "1": ("Surriscaldamento", "overheating"),
    "2": ("Vibrazione anomala", "abnormal_vibration"),
    "3": ("Anomalia della corrente", "current_anomaly"),
    "4": ("Anomalia della pressione", "pressure_anomaly"),
    "5": ("Anomalia della portata", "flow_anomaly"),
    "6": ("Rumore anomalo", "unusual_noise"),
    "7": ("Prestazioni ridotte", "reduced_performance"),
}

STATE_TRANSLATION = {
    "1": ("PRESENTE", "PRESENT"),
    "2": ("ASSENTE", "ABSENT"),
    "3": ("SCONOSCIUTO", "UNKNOWN"),
    "4": ("NON APPLICABILE", "NON_APPLICABLE"),
}


def initialize_system():
    """Costruisce l'ontologia e avvia il reasoner una sola volta."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Costruzione dell'ontologia OWL...")
    world, onto = build_ontology()

    print("\n[2/2] Avvio del reasoner HermiT...")
    reasoner = DiagnosticReasoner(onto)
    reasoner.run_inference()

    # Esporta l'ontologia dopo il reasoner, così il file salvato include
    # le classificazioni inferite disponibili nell'ontologia.
    ontology_path = RESULTS_DIR / "industrial_diagnosis.owl"
    onto.save(file=str(ontology_path), format="rdfxml")
    print(f"Ontologia salvata in: {ontology_path}")

    print("Sistema inizializzato correttamente.")
    return world, onto, reasoner


def show_ontology_summary(onto):
    """Mostra un riepilogo della base di conoscenza."""
    summary = ontology_summary(onto)

    print("\n=== RIEPILOGO ONTOLOGIA ===")
    print(f"Classi: {summary['classes']}")
    print(f"Proprietà tra individui: {summary['object_properties']}")
    print(f"Proprietà dei dati: {summary['data_properties']}")
    print(f"Individui: {summary['individuals']}")

    print("\nTipi di macchina:")
    for key, (label, _) in MACHINES.items():
        print(f"  {key}. {label}")

    print("\nFamiglie di guasto:")
    for name in (
        "ThermalFault",
        "MechanicalFault",
        "ElectricalFault",
        "HydraulicFault",
    ):
        fault_class = getattr(onto, name)
        print(f"  - {fault_class.label[0]}")


def run_symbolic_diagnosis(reasoner):
    """Acquisisce gli stati dei sintomi e mostra le ipotesi simboliche."""
    print("\n=== DIAGNOSI SIMBOLICA ===")
    print("Seleziona il tipo di macchina:")

    for key, (label, _) in MACHINES.items():
        print(f"  {key}. {label}")

    machine_choice = input("Scelta: ").strip()

    if machine_choice not in MACHINES:
        print("Scelta della macchina non valida.")
        return

    machine_label, machine_name = MACHINES[machine_choice]
    symptom_states = {}

    print("\nIndica lo stato di ogni sintomo:")
    print("  1. Presente")
    print("  2. Assente")
    print("  3. Sconosciuto")
    print("  4. Non applicabile")
    print("  Invio senza scelta = sconosciuto")

    for key, (label, symptom_name) in SYMPTOMS.items():
        choice = input(f"\n{label} [{key}]: ").strip()

        if not choice:
            choice = "3"

        if choice not in STATE_TRANSLATION:
            print("Valore non valido; il sintomo sarà considerato sconosciuto.")
            choice = "3"

        _, internal_state = STATE_TRANSLATION[choice]
        symptom_states[symptom_name] = internal_state

    try:
        result = reasoner.diagnose(machine_name, symptom_states)
    except (ValueError, RuntimeError) as error:
        print(f"Impossibile completare la diagnosi: {error}")
        return

    print(f"\n=== RISULTATO PER: {machine_label.upper()} ===")
    print("Metodo: ragionamento simbolico basato sull'ontologia.")
    print("Il supporto è un conteggio di evidenze, non una probabilità.")

    for position, diagnosis in enumerate(result["diagnoses"], start=1):
        print(f"\n{position}. {diagnosis['fault']}")
        print(f"   Supporto simbolico: {diagnosis['support_count']}")

        supporting = diagnosis["supporting_symptoms"]
        if supporting:
            print("   Sintomi favorevoli: " + ", ".join(supporting))
        else:
            print("   Sintomi favorevoli: nessuno")

        absent = diagnosis["absent_associated_symptoms"]
        if absent:
            print("   Sintomi associati dichiarati assenti: " + ", ".join(absent))

    if result["ignored_symptoms"]:
        ignored = [
            f"{item['symptom']} ({item['state']})"
            for item in result["ignored_symptoms"]
        ]
        print("\nEvidenze non conteggiate: " + ", ".join(ignored))


def run_ml_evaluation(reasoner):
    """Esegue e salva il confronto sperimentale Base vs OntoBK."""
    print("\n=== VALUTAZIONE MACHINE LEARNING ===")

    if not DATASET_PATH.exists():
        print(f"Dataset non trovato: {DATASET_PATH}")
        return

    data = pd.read_csv(DATASET_PATH, encoding="utf-8")

    if "fault" not in data.columns:
        print("Il dataset non contiene la colonna target 'fault'.")
        return

    X = data.drop(columns=["fault"]).copy()
    y = data["fault"].copy()

    print(f"Esempi: {len(data)}")
    print(f"Feature grezze: {X.shape[1]}")
    print("Configurazione: 5 fold × 3 ripetizioni.")
    print("L'esperimento può richiedere alcuni minuti.")

    results = compare_representations(
        dataframe=X,
        y=y,
        reasoner=reasoner,
        n_splits=5,
        n_repeats=3,
    )

    rows = []

    for result in results:
        row = {
            "model": result.model_name,
            "representation": result.representation,
            **result.metrics,
        }
        rows.append(row)

        matrix_path = RESULTS_DIR / (
            f"confusion_{result.model_name.lower().replace(' ', '_')}"
            f"_{result.representation.lower()}.csv"
        )

        pd.DataFrame(
            result.confusion_matrix,
            index=result.labels,
            columns=result.labels,
        ).to_csv(matrix_path, encoding="utf-8")

    results_df = pd.DataFrame(rows)
    results_df["accuracy_delta_ontobk_vs_base_pp"] = float("nan")

    for model_name in results_df["model"].unique():
        base_rows = results_df[
            (results_df["model"] == model_name)
            & (results_df["representation"] == "Base")
        ]
        ontobk_rows = results_df[
            (results_df["model"] == model_name)
            & (results_df["representation"] == "OntoBK")
        ]

        if not base_rows.empty and not ontobk_rows.empty:
            delta = (
                ontobk_rows.iloc[0]["accuracy"]
                - base_rows.iloc[0]["accuracy"]
            ) * 100

            results_df.loc[
                (results_df["model"] == model_name)
                & (results_df["representation"] == "OntoBK"),
                "accuracy_delta_ontobk_vs_base_pp",
            ] = delta

    results_path = RESULTS_DIR / "comparison_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8")

    columns = [
        "model",
        "representation",
        "accuracy",
        "accuracy_std",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_macro_std",
        "accuracy_delta_ontobk_vs_base_pp",
    ]

    print("\n=== RISULTATI ===")
    print(results_df[columns].round(4).to_string(index=False))
    print(f"\nRisultati salvati in: {results_path}")


def main():
    print("=" * 60)
    print(" ICON — INDUSTRIAL DIAGNOSIS")
    print("=" * 60)

    try:
        world, onto, reasoner = initialize_system()
    except Exception as error:
        print(f"\nErrore durante l'inizializzazione: {error}")
        return

    try:
        while True:
            print("\n=== MENU PRINCIPALE ===")
            print("1. Diagnosi simbolica")
            print("2. Valutazione Machine Learning")
            print("3. Riepilogo dell'ontologia")
            print("4. Esci")

            choice = input("\nSeleziona un'opzione: ").strip()

            if choice == "1":
                run_symbolic_diagnosis(reasoner)
            elif choice == "2":
                run_ml_evaluation(reasoner)
            elif choice == "3":
                show_ontology_summary(onto)
            elif choice == "4":
                print("Chiusura del sistema ICON.")
                break
            else:
                print("Opzione non valida. Scegli un numero da 1 a 4.")

    except (KeyboardInterrupt, EOFError):
        print("\nChiusura del sistema ICON.")
    finally:
        world.close()


if __name__ == "__main__":
    main()
