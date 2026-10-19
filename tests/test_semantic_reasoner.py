
"""Test del reasoner semantico OWL."""

import pytest

from src.ontology import build_ontology
from src.semantic_reasoner import DiagnosticReasoner


@pytest.fixture
def ontology_and_reasoner():
    world, onto = build_ontology()
    reasoner = DiagnosticReasoner(onto)
    return world, onto, reasoner


def test_inference_requires_reasoner_execution(ontology_and_reasoner):
    _, _, reasoner = ontology_and_reasoner

    with pytest.raises(RuntimeError):
        reasoner.get_inferred_classes_for_symptom("pressure_anomaly")


def test_hydraulic_symptom_is_inferred(ontology_and_reasoner):
    _, _, reasoner = ontology_and_reasoner

    reasoner.run_inference()

    inferred = reasoner.get_inferred_classes_for_symptom(
        "pressure_anomaly"
    )

    assert "HydraulicSymptom" in inferred


def test_flow_anomaly_is_hydraulic(ontology_and_reasoner):
    _, _, reasoner = ontology_and_reasoner

    reasoner.run_inference()

    inferred = reasoner.get_inferred_classes_for_symptom(
        "flow_anomaly"
    )

    assert "HydraulicSymptom" in inferred


def test_overheating_has_multiple_inferred_categories(
    ontology_and_reasoner,
):
    _, _, reasoner = ontology_and_reasoner

    reasoner.run_inference()

    inferred = reasoner.get_inferred_classes_for_symptom("overheating")

    assert "ThermalSymptom" in inferred
    assert "MechanicalSymptom" in inferred
    assert "ElectricalSymptom" in inferred


def test_unknown_symptom_returns_empty_list(ontology_and_reasoner):
    _, _, reasoner = ontology_and_reasoner

    reasoner.run_inference()

    assert reasoner.get_inferred_classes_for_symptom(
        "symptom_that_does_not_exist"
    ) == []


def test_results_are_cached(ontology_and_reasoner):
    _, _, reasoner = ontology_and_reasoner

    reasoner.run_inference()

    first = reasoner.get_inferred_classes_for_symptom(
        "pressure_anomaly"
    )
    second = reasoner.get_inferred_classes_for_symptom(
        "pressure_anomaly"
    )

    assert first == second
    assert first is not second
    assert "pressure_anomaly" in reasoner.cache_inferenze


def test_diagnose_requires_reasoner_execution(ontology_and_reasoner):
    _, _, reasoner = ontology_and_reasoner

    with pytest.raises(RuntimeError):
        reasoner.diagnose(
            "industrial_pump",
            {"pressure_anomaly": "PRESENT"},
        )


def test_diagnose_supports_hydraulic_fault(ontology_and_reasoner):
    _, _, reasoner = ontology_and_reasoner
    reasoner.run_inference()

    result = reasoner.diagnose(
        "industrial_pump",
        {
            "pressure_anomaly": "PRESENT",
            "flow_anomaly": "PRESENT",
            "overheating": "UNKNOWN",
        },
    )

    diagnoses = result["diagnoses"]
    hydraulic = next(
        item for item in diagnoses
        if item["fault"] == "GUASTO_IDRAULICO"
    )

    assert hydraulic["support_count"] == 2
    assert "pressure_anomaly" in hydraulic["supporting_symptoms"]
    assert "flow_anomaly" in hydraulic["supporting_symptoms"]
    assert result["ignored_symptoms"] == [
        {"symptom": "overheating", "state": "UNKNOWN"}
    ]


def test_diagnose_respects_machine_compatibility(ontology_and_reasoner):
    _, _, reasoner = ontology_and_reasoner
    reasoner.run_inference()

    result = reasoner.diagnose(
        "electric_motor",
        {
            "pressure_anomaly": "PRESENT",
            "current_anomaly": "PRESENT",
        },
    )

    fault_labels = {
        item["fault"] for item in result["diagnoses"]
    }

    assert "GUASTO_IDRAULICO" not in fault_labels
    assert "GUASTO_ELETTRICO" in fault_labels


def test_unknown_and_non_applicable_are_not_counted(
    ontology_and_reasoner,
):
    _, _, reasoner = ontology_and_reasoner
    reasoner.run_inference()

    result = reasoner.diagnose(
        "industrial_pump",
        {
            "pressure_anomaly": "UNKNOWN",
            "flow_anomaly": "NON_APPLICABLE",
        },
    )

    assert all(
        item["support_count"] == 0
        for item in result["diagnoses"]
    )
    assert len(result["ignored_symptoms"]) == 2


def test_diagnose_rejects_invalid_state(ontology_and_reasoner):
    _, _, reasoner = ontology_and_reasoner
    reasoner.run_inference()

    with pytest.raises(ValueError):
        reasoner.diagnose(
            "industrial_pump",
            {"pressure_anomaly": "MAYBE"},
        )
