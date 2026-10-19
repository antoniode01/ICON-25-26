
import numpy as np
import pandas as pd
import pytest

from src.ontoBK_learning import (
    SemanticFeatureExtractor,
    build_models,
    build_pipeline,
)


@pytest.fixture
def sample_dataframe():
    """Piccolo dataset artificiale per testare l'estrazione."""
    return pd.DataFrame({
        "machine_type": ["electric_motor", "industrial_pump", "compressor", "conveyor_belt"],
        "load_fraction": [0.5, 0.7, 0.6, 0.8],
        "operating_hours": [100.0, 200.0, 150.0, 300.0],
        "temperature_c": [70.0, 65.0, 80.0, 60.0],
        "current_a": [10.0, 12.0, 11.0, 9.0],
        "vibration_mm_s": [2.0, 3.0, 4.0, 2.5],
        "pressure_bar": [np.nan, 5.0, 6.0, np.nan],
        "flow_rate_l_min": [np.nan, 20.0, 25.0, np.nan],
        "temperature_c_status": ["DISPONIBILE"] * 4,
        "current_a_status": ["DISPONIBILE"] * 4,
        "vibration_mm_s_status": ["DISPONIBILE"] * 4,
        "pressure_bar_status": ["NON_APPLICABILE", "DISPONIBILE", "DISPONIBILE", "NON_APPLICABILE"],
        "flow_rate_l_min_status": ["NON_APPLICABILE", "DISPONIBILE", "DISPONIBILE", "NON_APPLICABILE"],
        "symptom_overheating": ["PRESENTE", "ASSENTE", "PRESENTE", "ASSENTE"],
        "symptom_abnormal_vibration": ["ASSENTE", "PRESENTE", "PRESENTE", "ASSENTE"],
        "symptom_current_anomaly": ["PRESENTE", "ASSENTE", "ASSENTE", "PRESENTE"],
        "symptom_pressure_anomaly": ["NON_APPLICABILE", "PRESENTE", "ASSENTE", "NON_APPLICABILE"],
        "symptom_flow_anomaly": ["NON_APPLICABILE", "PRESENTE", "ASSENTE", "NON_APPLICABILE"],
        "symptom_unusual_noise": ["ASSENTE", "PRESENTE", "PRESENTE", "ASSENTE"],
        "symptom_reduced_performance": ["PRESENTE", "PRESENTE", "ASSENTE", "ASSENTE"],
    })


def test_extractor_is_sklearn_compatible():
    extractor = SemanticFeatureExtractor(reasoner=None)

    assert hasattr(extractor, "get_params")
    assert hasattr(extractor, "fit_transform")


def test_extractor_generates_base_features(sample_dataframe):
    extractor = SemanticFeatureExtractor(reasoner=None)
    X_base, X_ontobk = extractor.fit_transform(sample_dataframe)

    assert len(X_base) == len(sample_dataframe)
    assert len(X_ontobk) == len(sample_dataframe)
    assert X_base.shape[1] > 0
    assert X_ontobk.shape[1] >= X_base.shape[1]
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_base.dtypes)


def test_extractor_preserves_feature_alignment(sample_dataframe):
    extractor = SemanticFeatureExtractor(reasoner=None)
    X_base, X_ontobk = extractor.fit_transform(sample_dataframe)

    assert X_base.index.equals(sample_dataframe.index)
    assert X_ontobk.index.equals(sample_dataframe.index)
    assert X_base.columns.is_unique
    assert X_ontobk.columns.is_unique


def test_models_and_pipelines_are_constructible():
    models = build_models()

    assert set(models) == {"Random Forest", "SVM RBF", "MLP"}

    for model in models.values():
        pipeline = build_pipeline(model)
        assert pipeline is not None


def test_transform_requires_fit(sample_dataframe):
    extractor = SemanticFeatureExtractor(reasoner=None)

    with pytest.raises(RuntimeError):
        extractor.transform(sample_dataframe)
