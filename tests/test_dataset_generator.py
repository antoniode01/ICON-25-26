
import pandas as pd
import pytest

from src.dataset_generator import FAULTS, generate_dataset


@pytest.fixture
def dataset():
    return generate_dataset(n_samples=300, random_state=42)


def test_dataset_has_expected_size_and_columns(dataset):
    assert len(dataset) == 300
    assert len(dataset.columns) == 22
    assert "fault" in dataset.columns
    assert "machine_type" in dataset.columns


def test_fault_classes_are_balanced(dataset):
    counts = dataset["fault"].value_counts()

    assert set(counts.index) == set(FAULTS)
    assert counts.nunique() == 1


def test_fault_labels_are_in_italian(dataset):
    assert set(dataset["fault"].unique()) == {
        "GUASTO_TERMICO",
        "GUASTO_MECCANICO",
        "GUASTO_ELETTRICO",
        "GUASTO_IDRAULICO",
    }


def test_machine_labels_are_in_italian(dataset):
    assert set(dataset["machine_type"].unique()).issubset(
        {
            "MOTORE_ELETTRICO",
            "POMPA_INDUSTRIALE",
            "COMPRESSORE",
            "NASTRO_TRASPORTATORE",
        }
    )


def test_case_ids_are_unique(dataset):
    assert dataset["case_id"].is_unique


def test_generation_is_reproducible():
    first = generate_dataset(n_samples=100, random_state=123)
    second = generate_dataset(n_samples=100, random_state=123)

    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_can_generate_different_data():
    first = generate_dataset(n_samples=100, random_state=1)
    second = generate_dataset(n_samples=100, random_state=2)

    assert not first.equals(second)


def test_hydraulic_faults_use_compatible_machines(dataset):
    hydraulic = dataset[dataset["fault"] == "GUASTO_IDRAULICO"]

    assert len(hydraulic) > 0
    assert hydraulic["machine_type"].isin(
        ["POMPA_INDUSTRIALE", "COMPRESSORE"]
    ).all()


def test_pressure_and_flow_are_not_applicable_to_other_machines(dataset):
    machines_without_pressure_flow = dataset[
        dataset["machine_type"].isin(
            ["MOTORE_ELETTRICO", "NASTRO_TRASPORTATORE"]
        )
    ]

    assert machines_without_pressure_flow["pressure_bar"].isna().all()
    assert machines_without_pressure_flow["flow_rate_l_min"].isna().all()

    assert (
        machines_without_pressure_flow["symptom_pressure_anomaly"]
        == "NON_APPLICABILE"
    ).all()

    assert (
        machines_without_pressure_flow["symptom_flow_anomaly"]
        == "NON_APPLICABILE"
    ).all()


@pytest.mark.parametrize(
    "sensor",
    [
        "temperature_c",
        "current_a",
        "vibration_mm_s",
        "pressure_bar",
        "flow_rate_l_min",
    ],
)
def test_sensor_values_match_their_status(dataset, sensor):
    values = dataset[sensor]
    statuses = dataset[f"{sensor}_status"]

    assert set(statuses.unique()).issubset(
        {"DISPONIBILE", "MANCANTE", "NON_APPLICABILE"}
    )

    assert values[statuses == "DISPONIBILE"].notna().all()
    assert values[statuses != "DISPONIBILE"].isna().all()


def test_sensor_readings_are_nonnegative(dataset):
    sensors = [
        "temperature_c",
        "current_a",
        "vibration_mm_s",
        "pressure_bar",
        "flow_rate_l_min",
    ]

    for sensor in sensors:
        assert (dataset[sensor].dropna() >= 0).all()


def test_symptoms_use_valid_italian_states(dataset):
    valid_states = {
        "PRESENTE",
        "ASSENTE",
        "SCONOSCIUTO",
        "NON_APPLICABILE",
    }

    symptom_columns = [
        column for column in dataset.columns
        if column.startswith("symptom_")
    ]

    for column in symptom_columns:
        assert set(dataset[column].unique()).issubset(valid_states)


def test_not_applicable_is_used_only_for_pressure_and_flow_symptoms(dataset):
    symptom_columns = [
        column for column in dataset.columns
        if column.startswith("symptom_")
    ]

    for column in symptom_columns:
        if column not in {
            "symptom_pressure_anomaly",
            "symptom_flow_anomaly",
        }:
            assert (dataset[column] != "NON_APPLICABILE").all()


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        generate_dataset(n_samples=0)

    with pytest.raises(ValueError):
        generate_dataset(unknown_probability=1.0)

    with pytest.raises(ValueError):
        generate_dataset(sensor_missing_probability=-0.1)
