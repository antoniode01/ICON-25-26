
"""Generatore di dati sintetici per la diagnosi dei guasti industriali."""

from pathlib import Path

import numpy as np
import pandas as pd


# Etichette pubbliche delle quattro classi di diagnosi.
FAULTS = (
    "GUASTO_TERMICO",
    "GUASTO_MECCANICO",
    "GUASTO_ELETTRICO",
    "GUASTO_IDRAULICO",
)

MACHINES_BY_FAULT = {
    "GUASTO_TERMICO": (
        "MOTORE_ELETTRICO",
        "POMPA_INDUSTRIALE",
        "COMPRESSORE",
        "NASTRO_TRASPORTATORE",
    ),
    "GUASTO_MECCANICO": (
        "MOTORE_ELETTRICO",
        "POMPA_INDUSTRIALE",
        "COMPRESSORE",
        "NASTRO_TRASPORTATORE",
    ),
    "GUASTO_ELETTRICO": (
        "MOTORE_ELETTRICO",
        "POMPA_INDUSTRIALE",
        "COMPRESSORE",
        "NASTRO_TRASPORTATORE",
    ),
    "GUASTO_IDRAULICO": (
        "POMPA_INDUSTRIALE",
        "COMPRESSORE",
    ),
}

BASELINES = {
    "MOTORE_ELETTRICO": {
        "temperature": 65.0,
        "current": 12.0,
        "vibration": 2.0,
        "pressure": np.nan,
        "flow_rate": np.nan,
    },
    "POMPA_INDUSTRIALE": {
        "temperature": 58.0,
        "current": 10.0,
        "vibration": 2.2,
        "pressure": 5.0,
        "flow_rate": 80.0,
    },
    "COMPRESSORE": {
        "temperature": 75.0,
        "current": 15.0,
        "vibration": 2.5,
        "pressure": 8.0,
        "flow_rate": 100.0,
    },
    "NASTRO_TRASPORTATORE": {
        "temperature": 50.0,
        "current": 8.0,
        "vibration": 1.8,
        "pressure": np.nan,
        "flow_rate": np.nan,
    },
}

FAULT_EFFECTS = {
    "GUASTO_TERMICO": {
        "temperature": 24.0,
        "current": 1.5,
        "vibration": 0.5,
        "pressure": 0.0,
        "flow_rate": -2.0,
    },
    "GUASTO_MECCANICO": {
        "temperature": 8.0,
        "current": 2.0,
        "vibration": 5.0,
        "pressure": -0.4,
        "flow_rate": -5.0,
    },
    "GUASTO_ELETTRICO": {
        "temperature": 10.0,
        "current": 5.0,
        "vibration": 1.0,
        "pressure": 0.0,
        "flow_rate": -1.0,
    },
    "GUASTO_IDRAULICO": {
        "temperature": 5.0,
        "current": 1.0,
        "vibration": 1.5,
        "pressure": -2.2,
        "flow_rate": -25.0,
    },
}

# Probabilità illustrative della simulazione, non misure industriali reali.
SYMPTOM_PROBABILITIES = {
    "overheating": {
        "GUASTO_TERMICO": 0.78,
        "GUASTO_MECCANICO": 0.30,
        "GUASTO_ELETTRICO": 0.38,
        "GUASTO_IDRAULICO": 0.20,
    },
    "abnormal_vibration": {
        "GUASTO_TERMICO": 0.25,
        "GUASTO_MECCANICO": 0.78,
        "GUASTO_ELETTRICO": 0.30,
        "GUASTO_IDRAULICO": 0.38,
    },
    "current_anomaly": {
        "GUASTO_TERMICO": 0.32,
        "GUASTO_MECCANICO": 0.35,
        "GUASTO_ELETTRICO": 0.80,
        "GUASTO_IDRAULICO": 0.28,
    },
    "pressure_anomaly": {
        "GUASTO_TERMICO": 0.12,
        "GUASTO_MECCANICO": 0.30,
        "GUASTO_ELETTRICO": 0.15,
        "GUASTO_IDRAULICO": 0.82,
    },
    "flow_anomaly": {
        "GUASTO_TERMICO": 0.18,
        "GUASTO_MECCANICO": 0.32,
        "GUASTO_ELETTRICO": 0.18,
        "GUASTO_IDRAULICO": 0.80,
    },
    "unusual_noise": {
        "GUASTO_TERMICO": 0.22,
        "GUASTO_MECCANICO": 0.72,
        "GUASTO_ELETTRICO": 0.30,
        "GUASTO_IDRAULICO": 0.42,
    },
    "reduced_performance": {
        "GUASTO_TERMICO": 0.45,
        "GUASTO_MECCANICO": 0.52,
        "GUASTO_ELETTRICO": 0.50,
        "GUASTO_IDRAULICO": 0.68,
    },
}

SYMPTOM_LABELS = {
    "overheating": "Surriscaldamento",
    "abnormal_vibration": "Vibrazione anomala",
    "current_anomaly": "Anomalia della corrente",
    "pressure_anomaly": "Anomalia della pressione",
    "flow_anomaly": "Anomalia della portata",
    "unusual_noise": "Rumore anomalo",
    "reduced_performance": "Prestazioni ridotte",
}

MACHINE_LABELS = {
    "MOTORE_ELETTRICO": "Motore elettrico",
    "POMPA_INDUSTRIALE": "Pompa industriale",
    "COMPRESSORE": "Compressore",
    "NASTRO_TRASPORTATORE": "Nastro trasportatore",
}


def _sensor_value(
    rng: np.random.Generator,
    baseline: float,
    effect: float,
    load: float,
    noise_sd: float,
) -> float:
    """Genera una misura influenzata da guasto, carico e rumore."""
    return float(
        baseline
        + effect
        + (load - 0.65) * noise_sd * 0.8
        + rng.normal(0.0, noise_sd)
    )


def generate_dataset(
    n_samples: int = 3000,
    random_state: int = 42,
    unknown_probability: float = 0.08,
    sensor_missing_probability: float = 0.04,
) -> pd.DataFrame:
    """Genera un dataset sintetico bilanciato e riproducibile."""
    if n_samples < len(FAULTS):
        raise ValueError(
            "Il numero di casi deve essere almeno pari al numero di guasti."
        )

    if not 0.0 <= unknown_probability < 1.0:
        raise ValueError(
            "La probabilità di stato sconosciuto deve appartenere a [0, 1)."
        )

    if not 0.0 <= sensor_missing_probability < 1.0:
        raise ValueError(
            "La probabilità di sensore mancante deve appartenere a [0, 1)."
        )

    rng = np.random.default_rng(random_state)

    # Assegna lo stesso numero di casi a ogni classe, per quanto possibile.
    labels = np.resize(np.array(FAULTS, dtype=object), n_samples)
    rng.shuffle(labels)

    rows = []

    sensor_specs = {
        "temperature_c": ("temperature", 8.0),
        "current_a": ("current", 2.0),
        "vibration_mm_s": ("vibration", 1.2),
        "pressure_bar": ("pressure", 0.8),
        "flow_rate_l_min": ("flow_rate", 8.0),
    }

    for index, fault in enumerate(labels):
        fault = str(fault)
        machine = str(rng.choice(MACHINES_BY_FAULT[fault]))
        baseline = BASELINES[machine]
        effects = FAULT_EFFECTS[fault]

        load = float(rng.uniform(0.30, 1.00))
        operating_hours = float(rng.uniform(100.0, 20000.0))

        row = {
            "case_id": f"CASO_{index + 1:05d}",
            "machine_type": machine,
            "load_fraction": round(load, 4),
            "operating_hours": round(operating_hours, 2),
        }

        for column, (sensor, noise_sd) in sensor_specs.items():
            base_value = baseline[sensor]
            status_column = f"{column}_status"

            if pd.isna(base_value):
                row[column] = np.nan
                row[status_column] = "NON_APPLICABILE"
                continue

            if rng.random() < sensor_missing_probability:
                row[column] = np.nan
                row[status_column] = "MANCANTE"
                continue

            value = _sensor_value(
                rng,
                float(base_value),
                float(effects[sensor]),
                load,
                noise_sd,
            )

            row[column] = round(max(0.0, value), 3)
            row[status_column] = "DISPONIBILE"

        for symptom, probabilities in SYMPTOM_PROBABILITIES.items():
            if (
                symptom in {"pressure_anomaly", "flow_anomaly"}
                and pd.isna(baseline["pressure"])
            ):
                state = "NON_APPLICABILE"
            elif rng.random() < unknown_probability:
                state = "SCONOSCIUTO"
            else:
                state = (
                    "PRESENTE"
                    if rng.random() < probabilities[fault]
                    else "ASSENTE"
                )

            row[f"symptom_{symptom}"] = state

        row["fault"] = fault
        rows.append(row)

    return pd.DataFrame(rows)


def save_dataset(
    output_path: str | Path = "data/industrial_faults.csv",
    n_samples: int = 3000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Genera il dataset e lo salva in formato CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset(
        n_samples=n_samples,
        random_state=random_state,
    )
    dataset.to_csv(output_path, index=False)
    return dataset


if __name__ == "__main__":
    dataset = save_dataset()

    print(f"Dataset generato: {len(dataset)} righe")
    print(f"Numero di colonne: {len(dataset.columns)}")
    print("\nDistribuzione dei guasti:")
    print(dataset["fault"].value_counts().sort_index())
    print("\nDistribuzione delle macchine:")
    print(dataset["machine_type"].value_counts().sort_index())
    print("\nPrime cinque righe:")
    print(dataset.head().to_string(index=False))
