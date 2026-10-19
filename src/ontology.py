
"""Ontologia OWL per la diagnosi dei guasti industriali."""

from owlready2 import (
    AnnotationProperty,
    DataProperty,
    FunctionalProperty,
    ObjectProperty,
    Thing,
    World,
    locstr,
)


ONTOLOGY_IRI = "https://example.org/icon/industrial-diagnosis.owl"


def build_ontology():
    """Costruisce e restituisce il mondo OWL e l'ontologia."""
    world = World()
    onto = world.get_ontology(ONTOLOGY_IRI)

    with onto:
        # ---------- Etichette ----------

        class label(AnnotationProperty):
            namespace = world.get_namespace(
                "http://www.w3.org/2000/01/rdf-schema#"
            )

        # ---------- Concetti principali ----------

        class Machine(Thing):
            pass

        class Component(Thing):
            pass

        class Sensor(Thing):
            pass

        class Observation(Thing):
            pass

        class Symptom(Thing):
            pass

        class Fault(Thing):
            pass

        class FaultFamily(Fault):
            pass

        class ThermalFault(FaultFamily):
            pass

        class MechanicalFault(FaultFamily):
            pass

        class ElectricalFault(FaultFamily):
            pass

        class HydraulicFault(FaultFamily):
            pass

        class MachineType(Machine):
            pass

        class ElectricMotor(MachineType):
            pass

        class IndustrialPump(MachineType):
            pass

        class Compressor(MachineType):
            pass

        class ConveyorBelt(MachineType):
            pass

        class TemperatureSensor(Sensor):
            pass

        class CurrentSensor(Sensor):
            pass

        class VibrationSensor(Sensor):
            pass

        class PressureSensor(Sensor):
            pass

        class FlowSensor(Sensor):
            pass

        # ---------- Proprietà tra individui ----------

        class has_component(ObjectProperty):
            domain = [Machine]
            range = [Component]

        class has_sensor(ObjectProperty):
            domain = [Machine]
            range = [Sensor]

        class monitors_component(ObjectProperty):
            domain = [Sensor]
            range = [Component]

        class observes_symptom(ObjectProperty):
            domain = [Observation]
            range = [Symptom]

        class concerns_machine(ObjectProperty):
            domain = [Observation]
            range = [Machine]

        class indicates_fault(ObjectProperty):
            domain = [Symptom]
            range = [FaultFamily]

        class affects_component(ObjectProperty):
            domain = [FaultFamily]
            range = [Component]

        class has_observation(ObjectProperty):
            domain = [Machine]
            range = [Observation]

        class compatible_with_machine(ObjectProperty):
            domain = [FaultFamily]
            range = [MachineType]

        # ---------- Classificazione semantica dei sintomi ----------

        class ThermalSymptom(Symptom):
            equivalent_to = [
                Symptom & indicates_fault.some(ThermalFault)
            ]

        class MechanicalSymptom(Symptom):
            equivalent_to = [
                Symptom & indicates_fault.some(MechanicalFault)
            ]

        class ElectricalSymptom(Symptom):
            equivalent_to = [
                Symptom & indicates_fault.some(ElectricalFault)
            ]

        class HydraulicSymptom(Symptom):
            equivalent_to = [
                Symptom & indicates_fault.some(HydraulicFault)
            ]

        # ---------- Proprietà dei dati ----------

        class observation_state(DataProperty, FunctionalProperty):
            domain = [Observation]
            range = [str]

        class measurement_value(DataProperty, FunctionalProperty):
            domain = [Observation]
            range = [float]

        class measurement_unit(DataProperty, FunctionalProperty):
            domain = [Observation]
            range = [str]

        # ---------- Componenti ----------

        class MotorComponent(Component):
            pass

        class Bearing(Component):
            pass

        class CoolingSystem(Component):
            pass

        class ElectricalSupply(Component):
            pass

        class HydraulicCircuit(Component):
            pass

        class BeltDrive(Component):
            pass

        # ---------- Individui delle famiglie di guasto ----------

        thermal_fault = ThermalFault("thermal_fault")
        mechanical_fault = MechanicalFault("mechanical_fault")
        electrical_fault = ElectricalFault("electrical_fault")
        hydraulic_fault = HydraulicFault("hydraulic_fault")

        # ---------- Individui dei tipi di macchina ----------

        electric_motor = ElectricMotor("electric_motor")
        industrial_pump = IndustrialPump("industrial_pump")
        compressor = Compressor("compressor")
        conveyor_belt = ConveyorBelt("conveyor_belt")

        # ---------- Individui dei componenti ----------

        motor_component = MotorComponent("motor_component")
        bearing = Bearing("bearing")
        cooling_system = CoolingSystem("cooling_system")
        electrical_supply = ElectricalSupply("electrical_supply")
        hydraulic_circuit = HydraulicCircuit("hydraulic_circuit")
        belt_drive = BeltDrive("belt_drive")

        # ---------- Individui dei sensori ----------

        temperature_sensor = TemperatureSensor("temperature_sensor")
        current_sensor = CurrentSensor("current_sensor")
        vibration_sensor = VibrationSensor("vibration_sensor")
        pressure_sensor = PressureSensor("pressure_sensor")
        flow_sensor = FlowSensor("flow_sensor")

        # ---------- Individui dei sintomi ----------

        overheating = Symptom("overheating")
        abnormal_vibration = Symptom("abnormal_vibration")
        current_anomaly = Symptom("current_anomaly")
        pressure_anomaly = Symptom("pressure_anomaly")
        flow_anomaly = Symptom("flow_anomaly")
        unusual_noise = Symptom("unusual_noise")
        reduced_performance = Symptom("reduced_performance")

        # ---------- Macchine e componenti ----------

        electric_motor.has_component = [
            motor_component,
            bearing,
            cooling_system,
            electrical_supply,
        ]

        industrial_pump.has_component = [
            motor_component,
            bearing,
            cooling_system,
            electrical_supply,
            hydraulic_circuit,
        ]

        compressor.has_component = [
            motor_component,
            bearing,
            cooling_system,
            electrical_supply,
            hydraulic_circuit,
        ]

        conveyor_belt.has_component = [
            motor_component,
            bearing,
            electrical_supply,
            belt_drive,
        ]

        # ---------- Macchine e sensori ----------

        for machine in [
            electric_motor,
            industrial_pump,
            compressor,
            conveyor_belt,
        ]:
            machine.has_sensor = [
                temperature_sensor,
                current_sensor,
                vibration_sensor,
            ]

        for machine in [industrial_pump, compressor]:
            machine.has_sensor = [
                temperature_sensor,
                current_sensor,
                vibration_sensor,
                pressure_sensor,
                flow_sensor,
            ]

        temperature_sensor.monitors_component = [
            motor_component,
            cooling_system,
        ]

        current_sensor.monitors_component = [
            motor_component,
            electrical_supply,
        ]

        vibration_sensor.monitors_component = [
            motor_component,
            bearing,
            belt_drive,
        ]

        pressure_sensor.monitors_component = [hydraulic_circuit]
        flow_sensor.monitors_component = [hydraulic_circuit]

        # ---------- Sintomi e famiglie di guasto ----------

        overheating.indicates_fault = [
            thermal_fault,
            mechanical_fault,
            electrical_fault,
        ]

        abnormal_vibration.indicates_fault = [
            mechanical_fault,
            thermal_fault,
            hydraulic_fault,
        ]

        current_anomaly.indicates_fault = [
            electrical_fault,
            thermal_fault,
            mechanical_fault,
        ]

        pressure_anomaly.indicates_fault = [hydraulic_fault]
        flow_anomaly.indicates_fault = [hydraulic_fault]

        unusual_noise.indicates_fault = [
            mechanical_fault,
            hydraulic_fault,
            electrical_fault,
        ]

        reduced_performance.indicates_fault = [
            thermal_fault,
            mechanical_fault,
            electrical_fault,
            hydraulic_fault,
        ]

        # ---------- Guasti e componenti interessati ----------

        thermal_fault.affects_component = [
            cooling_system,
            motor_component,
        ]

        mechanical_fault.affects_component = [
            bearing,
            motor_component,
        ]

        electrical_fault.affects_component = [
            electrical_supply,
            motor_component,
        ]

        hydraulic_fault.affects_component = [hydraulic_circuit]

        # ---------- Compatibilità tra guasti e macchine ----------

        all_machines = [
            electric_motor,
            industrial_pump,
            compressor,
            conveyor_belt,
        ]

        thermal_fault.compatible_with_machine = all_machines
        mechanical_fault.compatible_with_machine = all_machines
        electrical_fault.compatible_with_machine = all_machines

        hydraulic_fault.compatible_with_machine = [
            industrial_pump,
            compressor,
        ]

        # ---------- Etichette italiane delle classi ----------

        class_labels = {
            Machine: "Macchina",
            Component: "Componente",
            Sensor: "Sensore",
            Observation: "Osservazione",
            Symptom: "Sintomo",
            Fault: "Guasto",
            FaultFamily: "Famiglia di guasto",
            ThermalFault: "Guasto termico",
            MechanicalFault: "Guasto meccanico",
            ElectricalFault: "Guasto elettrico",
            HydraulicFault: "Guasto idraulico",
            ThermalSymptom: "Sintomo termico",
            MechanicalSymptom: "Sintomo meccanico",
            ElectricalSymptom: "Sintomo elettrico",
            HydraulicSymptom: "Sintomo idraulico",
            MachineType: "Tipo di macchina",
            ElectricMotor: "Motore elettrico",
            IndustrialPump: "Pompa industriale",
            Compressor: "Compressore",
            ConveyorBelt: "Nastro trasportatore",
            TemperatureSensor: "Sensore di temperatura",
            CurrentSensor: "Sensore di corrente",
            VibrationSensor: "Sensore di vibrazione",
            PressureSensor: "Sensore di pressione",
            FlowSensor: "Sensore di portata",
            MotorComponent: "Componente motore",
            Bearing: "Cuscinetto",
            CoolingSystem: "Sistema di raffreddamento",
            ElectricalSupply: "Alimentazione elettrica",
            HydraulicCircuit: "Circuito idraulico",
            BeltDrive: "Trasmissione a cinghia",
        }

        property_labels = {
            has_component: "ha componente",
            has_sensor: "ha sensore",
            monitors_component: "monitora componente",
            observes_symptom: "osserva sintomo",
            concerns_machine: "riguarda macchina",
            indicates_fault: "indica guasto",
            affects_component: "interessa componente",
            has_observation: "ha osservazione",
            compatible_with_machine: "compatibile con macchina",
            observation_state: "stato dell'osservazione",
            measurement_value: "valore della misurazione",
            measurement_unit: "unità di misura",
        }

        individual_labels = {
            thermal_fault: "Guasto termico",
            mechanical_fault: "Guasto meccanico",
            electrical_fault: "Guasto elettrico",
            hydraulic_fault: "Guasto idraulico",
            electric_motor: "Motore elettrico",
            industrial_pump: "Pompa industriale",
            compressor: "Compressore",
            conveyor_belt: "Nastro trasportatore",
            motor_component: "Componente motore",
            bearing: "Cuscinetto",
            cooling_system: "Sistema di raffreddamento",
            electrical_supply: "Alimentazione elettrica",
            hydraulic_circuit: "Circuito idraulico",
            belt_drive: "Trasmissione a cinghia",
            temperature_sensor: "Sensore di temperatura",
            current_sensor: "Sensore di corrente",
            vibration_sensor: "Sensore di vibrazione",
            pressure_sensor: "Sensore di pressione",
            flow_sensor: "Sensore di portata",
            overheating: "Surriscaldamento",
            abnormal_vibration: "Vibrazione anomala",
            current_anomaly: "Anomalia della corrente",
            pressure_anomaly: "Anomalia della pressione",
            flow_anomaly: "Anomalia della portata",
            unusual_noise: "Rumore anomalo",
            reduced_performance: "Prestazioni ridotte",
        }

        for entity, italian_label in class_labels.items():
            entity.label = [locstr(italian_label, "it")]

        for entity, italian_label in property_labels.items():
            entity.label = [locstr(italian_label, "it")]

        for entity, italian_label in individual_labels.items():
            entity.label = [locstr(italian_label, "it")]

    return world, onto


def ontology_summary(onto):
    """Restituisce un riepilogo della struttura dell'ontologia."""
    return {
        "classes": len(list(onto.classes())),
        "object_properties": len(list(onto.object_properties())),
        "data_properties": len(list(onto.data_properties())),
        "individuals": len(list(onto.individuals())),
    }


if __name__ == "__main__":
    world, onto = build_ontology()

    print("Ontologia per la diagnosi industriale creata.")
    print(ontology_summary(onto))

    print("\nFamiglie di guasto:")
    for fault_class in [
        onto.ThermalFault,
        onto.MechanicalFault,
        onto.ElectricalFault,
        onto.HydraulicFault,
    ]:
        print(f"- {fault_class.label[0]}")

    print("\nMacchine:")
    for machine in [
        onto.electric_motor,
        onto.industrial_pump,
        onto.compressor,
        onto.conveyor_belt,
    ]:
        print(
            f"- {machine.label[0]}: "
            f"{len(machine.has_component)} componenti"
        )
