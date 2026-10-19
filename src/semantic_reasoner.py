
"""Reasoner semantico OWL per la diagnosi dei guasti industriali."""

from owlready2 import sync_reasoner


class DiagnosticReasoner:
    """Esegue inferenze OWL e supporta la diagnosi simbolica."""

    EXCLUDED_CLASSES = {
        "Thing",
        "Symptom",
        "Fault",
        "FaultFamily",
    }

    # Etichette del dataset associate agli individui OWL dei guasti.
    FAULTS = {
        "GUASTO_TERMICO": "thermal_fault",
        "GUASTO_MECCANICO": "mechanical_fault",
        "GUASTO_ELETTRICO": "electrical_fault",
        "GUASTO_IDRAULICO": "hydraulic_fault",
    }

    # Nomi degli individui OWL dei sintomi.
    SYMPTOMS = {
        "overheating",
        "abnormal_vibration",
        "current_anomaly",
        "pressure_anomaly",
        "flow_anomaly",
        "unusual_noise",
        "reduced_performance",
    }

    VALID_STATES = {
        "PRESENT",
        "ABSENT",
        "UNKNOWN",
        "NON_APPLICABLE",
    }

    def __init__(self, ontology):
        self.onto = ontology
        self.reasoner_run = False
        self.cache_inferenze = {}

    def run_inference(self):
        """Esegue HermiT e aggiorna la cache delle inferenze."""
        self.reasoner_run = False
        self.cache_inferenze.clear()

        print("[Reasoner] Avvio del reasoner OWL HermiT...")

        with self.onto:
            sync_reasoner(
                self.onto.world,
                infer_property_values=True,
            )

        self.reasoner_run = True
        print("[Reasoner] Inferenza completata.")

    def _get_individual(self, name):
        """Recupera un individuo OWL tramite il suo nome locale."""
        return self.onto.search_one(iri="*" + name)

    def get_inferred_classes_for_symptom(self, symptom_name):
        """
        Restituisce le classi inferite per un individuo sintomo.
        Richiede una precedente esecuzione di run_inference().
        """
        if not self.reasoner_run:
            raise RuntimeError(
                "Eseguire run_inference() prima di interrogare "
                "il reasoner."
            )

        if symptom_name in self.cache_inferenze:
            return list(self.cache_inferenze[symptom_name])

        symptom = self._get_individual(symptom_name)

        if symptom is None:
            return []

        is_symptom = any(
            getattr(cls, "name", None) == "Symptom"
            for cls in symptom.INDIRECT_is_a
        )

        if not is_symptom:
            return []

        inferred = sorted({
            cls.name
            for cls in symptom.INDIRECT_is_a
            if hasattr(cls, "name")
            and cls.name not in self.EXCLUDED_CLASSES
        })

        self.cache_inferenze[symptom_name] = inferred
        return list(inferred)

    def diagnose(self, machine_name, symptom_states):
        """
        Produce una diagnosi simbolica a partire dalla macchina
        e dagli stati osservati dei sintomi.

        Args:
            machine_name: nome dell'individuo macchina OWL,
                ad esempio 'industrial_pump'.
            symptom_states: dizionario del tipo
                {'overheating': 'PRESENT',
                 'pressure_anomaly': 'UNKNOWN'}.

        Returns:
            Dizionario con ipotesi ordinate, evidenze e informazioni
            non utilizzate nella diagnosi.

        Nota:
            Il numero di sintomi favorevoli è un conteggio simbolico,
            non una probabilità di guasto.
        """
        if not self.reasoner_run:
            raise RuntimeError(
                "Eseguire run_inference() prima della diagnosi."
            )

        machine = self._get_individual(machine_name)

        if machine is None:
            raise ValueError(
                f"Macchina OWL non trovata: {machine_name}"
            )

        if not isinstance(symptom_states, dict):
            raise TypeError(
                "symptom_states deve essere un dizionario."
            )

        normalized_states = {}

        for symptom_name, state in symptom_states.items():
            if symptom_name not in self.SYMPTOMS:
                raise ValueError(
                    f"Sintomo non riconosciuto: {symptom_name}"
                )

            if not isinstance(state, str):
                raise ValueError(
                    f"Stato non valido per {symptom_name}: {state}"
                )

            state = state.strip().upper()

            if state not in self.VALID_STATES:
                raise ValueError(
                    f"Stato non valido per {symptom_name}: {state}"
                )

            normalized_states[symptom_name] = state

        # Recupera i guasti compatibili con la macchina.
        compatible_faults = {}

        for label, individual_name in self.FAULTS.items():
            fault = self._get_individual(individual_name)

            if fault is None:
                raise ValueError(
                    f"Individuo OWL del guasto non trovato: "
                    f"{individual_name}"
                )

            compatible_machines = list(
                getattr(fault, "compatible_with_machine", [])
            )

            # Non si presume compatibilità se la relazione manca.
            if machine in compatible_machines:
                compatible_faults[label] = fault

        if not compatible_faults:
            raise ValueError(
                f"Nessun guasto compatibile definito nell'ontologia "
                f"per la macchina '{machine_name}'."
            )

        # Ogni sintomo presente fornisce supporto ai guasti
        # collegati tramite la relazione OWL indicates_fault.
        diagnoses = {}

        for label, fault in compatible_faults.items():
            diagnoses[label] = {
                "fault_individual": fault.name,
                "support_count": 0,
                "supporting_symptoms": [],
                "absent_associated_symptoms": [],
            }

        ignored_symptoms = []

        for symptom_name, state in normalized_states.items():
            if state in {"UNKNOWN", "NON_APPLICABLE"}:
                ignored_symptoms.append({
                    "symptom": symptom_name,
                    "state": state,
                })
                continue

            symptom = self._get_individual(symptom_name)

            if symptom is None:
                raise ValueError(
                    f"Individuo OWL del sintomo non trovato: "
                    f"{symptom_name}"
                )

            indicated_faults = set(
                getattr(symptom, "indicates_fault", [])
            )

            for label, fault in compatible_faults.items():
                if fault not in indicated_faults:
                    continue

                if state == "PRESENT":
                    diagnoses[label]["support_count"] += 1
                    diagnoses[label]["supporting_symptoms"].append(
                        symptom_name
                    )
                elif state == "ABSENT":
                    diagnoses[label][
                        "absent_associated_symptoms"
                    ].append(symptom_name)

        # Ordina per numero di sintomi presenti che supportano
        # l'ipotesi. A parità, mantiene un ordine deterministico.
        ranked_diagnoses = sorted(
            [
                {
                    "fault": label,
                    **details,
                }
                for label, details in diagnoses.items()
            ],
            key=lambda item: (
                -item["support_count"],
                item["fault"],
            ),
        )

        return {
            "machine": machine_name,
            "diagnoses": ranked_diagnoses,
            "ignored_symptoms": ignored_symptoms,
            "method": "OWL relations and symbolic evidence counting",
        }
