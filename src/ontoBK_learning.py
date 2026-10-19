
"""Estrazione delle feature e valutazione ML per ICON — Industrial Diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


NUMERIC_FEATURES = [
    "load_fraction",
    "operating_hours",
    "temperature_c",
    "current_a",
    "vibration_mm_s",
    "pressure_bar",
    "flow_rate_l_min",
]

SENSOR_STATUS_COLUMNS = [
    "temperature_c_status",
    "current_a_status",
    "vibration_mm_s_status",
    "pressure_bar_status",
    "flow_rate_l_min_status",
]

SYMPTOM_COLUMNS = [
    "symptom_overheating",
    "symptom_abnormal_vibration",
    "symptom_current_anomaly",
    "symptom_pressure_anomaly",
    "symptom_flow_anomaly",
    "symptom_unusual_noise",
    "symptom_reduced_performance",
]

SYMPTOM_TO_ONTOLOGY = {
    "symptom_overheating": "overheating",
    "symptom_abnormal_vibration": "abnormal_vibration",
    "symptom_current_anomaly": "current_anomaly",
    "symptom_pressure_anomaly": "pressure_anomaly",
    "symptom_flow_anomaly": "flow_anomaly",
    "symptom_unusual_noise": "unusual_noise",
    "symptom_reduced_performance": "reduced_performance",
}

MACHINE_TO_ONTOLOGY = {
    "MOTORE_ELETTRICO": "electric_motor",
    "POMPA_INDUSTRIALE": "industrial_pump",
    "COMPRESSORE": "compressor",
    "NASTRO_TRASPORTATORE": "conveyor_belt",
}
SENSOR_STATES = (
    "DISPONIBILE",
    "MANCANTE",
    "SCONOSCIUTO",
    "NON_APPLICABILE",
)

SYMPTOM_STATES = (
    "PRESENTE",
    "ASSENTE",
    "SCONOSCIUTO",
    "NON_APPLICABILE",
)


class SemanticFeatureExtractor(BaseEstimator, TransformerMixin):
    """Costruisce le rappresentazioni Base e OntoBK dei casi industriali."""

    def __init__(self, reasoner):
        self.reasoner = reasoner
        self.machine_types: list[str] = []
        self.semantic_classes_vocab: list[str] = []
        self.feature_names_base: list[str] = []
        self.feature_names_ontobk: list[str] = []
        self._fitted = False
        self.fault_labels: list[str] = []
        self.fault_individuals: dict[str, Any] = {}
        self.symptom_faults: dict[str, set[str]] = {}
        self.machine_compatibility: dict[str, set[str]] = {}

    @staticmethod
    def _validate_dataframe(dataframe: pd.DataFrame) -> None:
        required = (
            NUMERIC_FEATURES
            + SENSOR_STATUS_COLUMNS
            + SYMPTOM_COLUMNS
            + ["machine_type"]
        )
        missing = [
            column for column in required
            if column not in dataframe.columns
        ]
        if missing:
            raise ValueError(
                "Colonne mancanti nel dataset: " + ", ".join(missing)
            )

    @staticmethod
    def _normalise_state(
        value: Any, allowed: tuple[str, ...]
    ) -> str:
        if pd.isna(value):
            return "SCONOSCIUTO"

        state = str(value).strip().upper()
        return state if state in allowed else "SCONOSCIUTO"

    
    def fit(self, dataframe: pd.DataFrame) -> "SemanticFeatureExtractor":
        """Apprende il vocabolario dai dati ricevuti, senza usare le etichette."""
        self._validate_dataframe(dataframe)

        self.machine_types = sorted(
            dataframe["machine_type"]
            .fillna("SCONOSCIUTO")
            .astype(str)
            .unique()
        )

        self.feature_names_base = (
            NUMERIC_FEATURES.copy()
            + [f"machine__{name}" for name in self.machine_types]
            + [
                f"{column}__{state}"
                for column in SENSOR_STATUS_COLUMNS
                for state in SENSOR_STATES
            ]
            + [
                f"{column}__{state}"
                for column in SYMPTOM_COLUMNS
                for state in SYMPTOM_STATES
            ]
        )

        semantic_classes: set[str] = set()

        if self.reasoner is not None:
            if not getattr(self.reasoner, "reasoner_run", False):
                raise RuntimeError(
                    "Il reasoner deve eseguire run_inference() prima del fit."
                )

            for symptom_name in SYMPTOM_TO_ONTOLOGY.values():
                semantic_classes.update(
                    self.reasoner.get_inferred_classes_for_symptom(
                        symptom_name
                    )
                )

        self.semantic_classes_vocab = sorted(semantic_classes)

        # Recupera le relazioni simboliche direttamente dall'ontologia.
        self.fault_labels = []
        self.fault_individuals = {}
        self.symptom_faults = {}
        self.machine_compatibility = {}

        if self.reasoner is not None:
            for label, individual_name in self.reasoner.FAULTS.items():
                fault = self.reasoner.onto.search_one(
                    iri="*" + individual_name
                )

                if fault is None:
                    raise ValueError(
                        f"Individuo OWL del guasto non trovato: "
                        f"{individual_name}"
                    )

                self.fault_labels.append(label)
                self.fault_individuals[label] = fault

                self.machine_compatibility[label] = {
                    machine.name
                    for machine in getattr(
                        fault, "compatible_with_machine", []
                    )
                }

            for symptom_name in SYMPTOM_TO_ONTOLOGY.values():
                symptom = self.reasoner.onto.search_one(
                    iri="*" + symptom_name
                )

                if symptom is None:
                    raise ValueError(
                        f"Individuo OWL del sintomo non trovato: "
                        f"{symptom_name}"
                    )

                self.symptom_faults[symptom_name] = {
                    fault.name
                    for fault in getattr(symptom, "indicates_fault", [])
                }

        self.fault_labels.sort()

        self.feature_names_ontobk = (
            self.feature_names_base
            + [
                f"onto__{name}"
                for name in self.semantic_classes_vocab
            ]
            + [
                f"onto__support__{label}"
                for label in self.fault_labels
            ]
            + [
                f"onto__absent__{label}"
                for label in self.fault_labels
            ]
            + [
                f"onto__compatible__{label}"
                for label in self.fault_labels
            ]
            + [
                f"onto__compatible_support__{label}"
                for label in self.fault_labels
            ]
        )

        self._fitted = True
        return self


    def transform(
        self, dataframe: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Trasforma i casi usando il vocabolario appreso in fit()."""
        if not self._fitted:
            raise RuntimeError("Chiamare fit() prima di transform().")

        self._validate_dataframe(dataframe)

        base = pd.DataFrame(index=dataframe.index)

        for column in NUMERIC_FEATURES:
            base[column] = pd.to_numeric(
                dataframe[column], errors="coerce"
            ).astype(float)

        machine = (
            dataframe["machine_type"]
            .fillna("SCONOSCIUTO")
            .astype(str)
        )

        for machine_type in self.machine_types:
            base[f"machine__{machine_type}"] = (
                machine == machine_type
            ).astype(float)

        for column in SENSOR_STATUS_COLUMNS:
            states = dataframe[column].map(
                lambda value: self._normalise_state(
                    value, SENSOR_STATES
                )
            )
            for state in SENSOR_STATES:
                base[f"{column}__{state}"] = (
                    states == state
                ).astype(float)

        for column in SYMPTOM_COLUMNS:
            states = dataframe[column].map(
                lambda value: self._normalise_state(
                    value, SYMPTOM_STATES
                )
            )
            for state in SYMPTOM_STATES:
                base[f"{column}__{state}"] = (
                    states == state
                ).astype(float)

        # I tipi di macchina non osservati durante fit producono zeri
        # nelle colonne machine__; le altre feature restano disponibili.
        base = base.reindex(
            columns=self.feature_names_base,
            fill_value=0.0,
        )

        enriched = base.copy()

        for class_name in self.semantic_classes_vocab:
            enriched[f"onto__{class_name}"] = 0.0

        
        if self.reasoner is not None:
            for symptom_column, individual_name in (
                SYMPTOM_TO_ONTOLOGY.items()
            ):
                present = (
                    dataframe[symptom_column].map(
                        lambda value: self._normalise_state(
                            value, SYMPTOM_STATES
                        )
                    )
                    == "PRESENTE"
                )

                if not present.any():
                    continue

                inferred_classes = (
                    self.reasoner.get_inferred_classes_for_symptom(
                        individual_name
                    )
                )

                for class_name in inferred_classes:
                    feature = f"onto__{class_name}"
                    if feature in enriched.columns:
                        enriched.loc[present, feature] = 1.0

                
        # Feature simboliche derivate dalle relazioni OWL.
        for label in self.fault_labels:
            enriched[f"onto__support__{label}"] = 0.0
            enriched[f"onto__absent__{label}"] = 0.0
            enriched[f"onto__compatible__{label}"] = 0.0
            enriched[f"onto__compatible_support__{label}"] = 0.0

        for position, (_, row) in enumerate(dataframe.iterrows()):
            machine_name = MACHINE_TO_ONTOLOGY.get(
                str(row["machine_type"]).strip().upper()
            )

            for label in self.fault_labels:
                fault_name = self.fault_individuals[label].name

                compatible = float(
                    machine_name is not None
                    and machine_name
                    in self.machine_compatibility[label]
                )

                support_count = 0
                absent_count = 0

                for symptom_column, symptom_name in (
                    SYMPTOM_TO_ONTOLOGY.items()
                ):
                    state = self._normalise_state(
                        row[symptom_column], SYMPTOM_STATES
                    )

                    if (
                        fault_name
                        not in self.symptom_faults[symptom_name]
                    ):
                        continue

                    if state == "PRESENTE":
                        support_count += 1
                    elif state == "ASSENTE":
                        absent_count += 1

                enriched.iloc[
                    position,
                    enriched.columns.get_loc(
                        f"onto__support__{label}"
                    ),
                ] = support_count

                enriched.iloc[
                    position,
                    enriched.columns.get_loc(
                        f"onto__absent__{label}"
                    ),
                ] = absent_count

                enriched.iloc[
                    position,
                    enriched.columns.get_loc(
                        f"onto__compatible__{label}"
                    ),
                ] = compatible

                enriched.iloc[
                    position,
                    enriched.columns.get_loc(
                        f"onto__compatible_support__{label}"
                    ),
                ] = support_count * compatible

        enriched = enriched.reindex(
            columns=self.feature_names_ontobk,
            fill_value=0.0,
        )

        return base, enriched

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
        y: Any = None,
        **fit_params: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Adatta l'estrattore e restituisce Base e OntoBK."""
        return self.fit(dataframe).transform(dataframe)

    def process_dataset(
        self, dataframe: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Utility esplorativa: non usare prima della valutazione."""
        if "fault" not in dataframe.columns:
            raise ValueError("Il dataset deve contenere la colonna 'fault'.")

        features = dataframe.drop(columns=["fault"])
        X_base, X_ontobk = self.fit_transform(features)

        return X_base, X_ontobk, dataframe["fault"].copy()


def build_models() -> dict[str, Any]:
    """Restituisce i tre classificatori del confronto sperimentale."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
        "SVM RBF": SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=42,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(50, 25),
            max_iter=500,
            early_stopping=True,
            random_state=42,
        ),
    }


def build_pipeline(model: Any, scale: bool = True) -> Pipeline:
    """Imputa e scala le feature all'interno di ciascun fold."""
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "scaler",
                StandardScaler() if scale else "passthrough",
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, slice(0, None)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", clone(model)),
        ]
    )


@dataclass
class EvaluationResult:
    """Risultati aggregati della valutazione di un modello."""

    representation: str
    model_name: str
    metrics: dict[str, float]
    confusion_matrix: np.ndarray
    labels: list[str]


def compare_representations(
    dataframe: pd.DataFrame,
    y: pd.Series,
    reasoner: Any,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> list[EvaluationResult]:
    """Confronta Base e OntoBK apprendendo le feature dentro ogni fold.

    Le metriche sono medie dei singoli fold. La matrice di confusione aggrega
    le predizioni out-of-fold di tutte le ripetizioni: ogni caso viene quindi
    conteggiato una volta per ripetizione.
    """
    if len(dataframe) != len(y):
        raise ValueError(
            "dataframe e y devono avere la stessa lunghezza."
        )

    if not dataframe.index.equals(y.index):
        raise ValueError(
            "Gli indici di dataframe e y devono coincidere."
        )

    if n_splits < 2 or n_repeats < 1:
        raise ValueError(
            "Servono almeno 2 fold e 1 ripetizione."
        )

    if y.isna().any():
        raise ValueError("La colonna target y contiene valori mancanti.")

    class_counts = y.value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            "La valutazione richiede almeno due classi."
        )

    if class_counts.min() < n_splits:
        raise ValueError(
            "Ogni classe deve avere almeno n_splits esempi "
            "per la cross-validation."
        )

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=42,
    )
    splits = list(cv.split(dataframe, y))

    models = build_models()
    results: list[EvaluationResult] = []
    labels = sorted(y.astype(str).unique().tolist())

    for model_name, model in models.items():
        for representation in ("Base", "OntoBK"):
            fold_metrics: dict[str, list[float]] = {
                "accuracy": [],
                "precision_macro": [],
                "recall_macro": [],
                "f1_macro": [],
            }

            all_true: list[str] = []
            all_predictions: list[str] = []

            print(
                f"[Valutazione] {model_name} — {representation}"
            )

            for train_idx, test_idx in splits:
                train_data = dataframe.iloc[train_idx]
                test_data = dataframe.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                # Il vocabolario delle feature viene appreso soltanto
                # sul training fold.
                extractor = SemanticFeatureExtractor(reasoner)

                X_train_base, X_train_ontobk = (
                    extractor.fit_transform(train_data)
                )
                X_test_base, X_test_ontobk = (
                    extractor.transform(test_data)
                )

                if representation == "Base":
                    X_train = X_train_base
                    X_test = X_test_base
                else:
                    X_train = X_train_ontobk
                    X_test = X_test_ontobk

                pipeline = build_pipeline(model)
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)

                fold_metrics["accuracy"].append(
                    accuracy_score(y_test, predictions)
                )
                fold_metrics["precision_macro"].append(
                    precision_score(
                        y_test,
                        predictions,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                )
                fold_metrics["recall_macro"].append(
                    recall_score(
                        y_test,
                        predictions,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                )
                fold_metrics["f1_macro"].append(
                    f1_score(
                        y_test,
                        predictions,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                )

                all_true.extend(y_test.astype(str).tolist())
                all_predictions.extend(
                    pd.Series(predictions).astype(str).tolist()
                )

            metrics: dict[str, float] = {}

            for metric, values in fold_metrics.items():
                metrics[metric] = float(np.mean(values))
                metrics[f"{metric}_std"] = float(np.std(values))

            matrix = confusion_matrix(
                all_true,
                all_predictions,
                labels=labels,
            )

            result = EvaluationResult(
                representation=representation,
                model_name=model_name,
                metrics=metrics,
                confusion_matrix=matrix,
                labels=labels,
            )
            results.append(result)

            print(
                f"  Accuracy: {metrics['accuracy']:.4f} "
                f"± {metrics['accuracy_std']:.4f}"
            )
            print(
                f"  F1 macro: {metrics['f1_macro']:.4f} "
                f"± {metrics['f1_macro_std']:.4f}"
            )

    return results
