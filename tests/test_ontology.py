
import pytest

from src.ontology import build_ontology, ontology_summary


@pytest.fixture
def ontology():
    _, onto = build_ontology()
    return onto


def test_ontology_contains_expected_structure(ontology):
    summary = ontology_summary(ontology)

    assert summary["classes"] >= 20
    assert summary["object_properties"] >= 8
    assert summary["data_properties"] >= 3
    assert summary["individuals"] >= 20


@pytest.mark.parametrize(
    "class_name",
    [
        "Machine",
        "Component",
        "Sensor",
        "Observation",
        "Symptom",
        "Fault",
        "FaultFamily",
        "ThermalFault",
        "MechanicalFault",
        "ElectricalFault",
        "HydraulicFault",
    ],
)
def test_required_classes_exist(ontology, class_name):
    assert getattr(ontology, class_name, None) is not None


def test_four_machine_types_exist(ontology):
    assert ontology.electric_motor is not None
    assert ontology.industrial_pump is not None
    assert ontology.compressor is not None
    assert ontology.conveyor_belt is not None


def test_each_machine_has_components(ontology):
    machines = [
        ontology.electric_motor,
        ontology.industrial_pump,
        ontology.compressor,
        ontology.conveyor_belt,
    ]

    for machine in machines:
        assert len(machine.has_component) > 0


def test_hydraulic_fault_is_compatible_only_with_supported_machines(ontology):
    compatible = set(ontology.hydraulic_fault.compatible_with_machine)

    assert compatible == {
        ontology.industrial_pump,
        ontology.compressor,
    }


def test_pressure_and_flow_sensors_are_assigned_correctly(ontology):
    machines_without_hydraulic_sensors = [
        ontology.electric_motor,
        ontology.conveyor_belt,
    ]

    for machine in machines_without_hydraulic_sensors:
        assert ontology.pressure_sensor not in machine.has_sensor
        assert ontology.flow_sensor not in machine.has_sensor

    for machine in [ontology.industrial_pump, ontology.compressor]:
        assert ontology.pressure_sensor in machine.has_sensor
        assert ontology.flow_sensor in machine.has_sensor


def test_symptoms_link_to_expected_faults(ontology):
    assert ontology.thermal_fault in ontology.overheating.indicates_fault
    assert (
        ontology.mechanical_fault
        in ontology.abnormal_vibration.indicates_fault
    )
    assert ontology.electrical_fault in ontology.current_anomaly.indicates_fault
    assert ontology.hydraulic_fault in ontology.pressure_anomaly.indicates_fault
    assert ontology.hydraulic_fault in ontology.flow_anomaly.indicates_fault


def test_faults_link_to_affected_components(ontology):
    assert (
        ontology.cooling_system
        in ontology.thermal_fault.affects_component
    )
    assert ontology.bearing in ontology.mechanical_fault.affects_component
    assert (
        ontology.electrical_supply
        in ontology.electrical_fault.affects_component
    )
    assert (
        ontology.hydraulic_circuit
        in ontology.hydraulic_fault.affects_component
    )


def test_classes_have_italian_labels(ontology):
    expected_labels = {
        "Machine": "Macchina",
        "Component": "Componente",
        "Sensor": "Sensore",
        "Symptom": "Sintomo",
        "ThermalFault": "Guasto termico",
        "MechanicalFault": "Guasto meccanico",
        "ElectricalFault": "Guasto elettrico",
        "HydraulicFault": "Guasto idraulico",
    }

    for class_name, expected_label in expected_labels.items():
        cls = getattr(ontology, class_name)
        assert expected_label in [str(label) for label in cls.label]


def test_machine_individuals_have_italian_labels(ontology):
    expected_labels = {
        "electric_motor": "Motore elettrico",
        "industrial_pump": "Pompa industriale",
        "compressor": "Compressore",
        "conveyor_belt": "Nastro trasportatore",
    }

    for individual_name, expected_label in expected_labels.items():
        individual = getattr(ontology, individual_name)
        assert expected_label in [
            str(label) for label in individual.label
        ]
