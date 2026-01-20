"""Integration tests for the full pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_data
from src.data.preprocessor import DataPreprocessor
from src.models.predictor import StressPredictor
from src.models.trainer import ModelTrainer


class TestTrainingPipeline:
    """Integration tests for the training pipeline."""

    def test_full_training_pipeline(self, sample_dataframe: pd.DataFrame) -> None:
        """Test the complete training pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            sample_dataframe.to_csv(csv_path, index=False)

            df = load_data(csv_path)
            assert len(df) == len(sample_dataframe)

            preprocessor = DataPreprocessor()
            df_encoded = preprocessor.transform_dataframe(df)
            assert df_encoded["Experience_Years"].dtype in [np.int64, np.int32]

            trainer = ModelTrainer()
            X_train, X_test, y_train, y_test = trainer.prepare_data(df)

            assert len(X_train) + len(X_test) == len(df)

            trainer.train(X_train, y_train)
            assert trainer.model is not None

            metrics = trainer.evaluate(X_train, X_test, y_train, y_test)
            assert "test_r2" in metrics
            assert "test_mse" in metrics
            # R² can be negative with random data, so we just check it's computed
            assert isinstance(metrics["test_r2"], float)

            model_path = Path(tmpdir) / "model.joblib"
            trainer.save_model(model_path)
            assert model_path.exists()

            predictor = StressPredictor(model_path=model_path)
            predictor.load()

            sample_input = {
                "Hours_Worked": 10,
                "Sleep_Hours": 6,
                "Bugs": 15,
                "Deadline_Days": 7,
                "Coffee_Cups": 4,
                "Meetings": 3,
                "Interruptions": 5,
                "Experience_Years": "Mid",
                "Code_Complexity": "Medium",
                "Remote_Work": "Yes",
            }

            result = predictor.predict(sample_input)
            assert "stress_level" in result
            assert 0 <= result["stress_level"] <= 100

    def test_model_reproducibility(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that training is reproducible with same random state."""
        trainer1 = ModelTrainer({"n_estimators": 50, "random_state": 42})
        trainer2 = ModelTrainer({"n_estimators": 50, "random_state": 42})

        X_train, X_test, y_train, y_test = trainer1.prepare_data(sample_dataframe)

        trainer1.train(X_train, y_train)
        trainer2.train(X_train, y_train)

        pred1 = trainer1.model.predict(X_test)
        pred2 = trainer2.model.predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)

    def test_feature_importance_consistency(self, sample_dataframe: pd.DataFrame) -> None:
        """Test feature importance is consistent between trainer and predictor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"

            trainer = ModelTrainer()
            X_train, X_test, y_train, y_test = trainer.prepare_data(sample_dataframe)
            trainer.train(X_train, y_train)
            trainer.evaluate(X_train, X_test, y_train, y_test)
            trainer.save_model(model_path)

            trainer_importance = trainer.get_feature_importance()

            predictor = StressPredictor(model_path=model_path)
            predictor.load()
            predictor_importance = predictor.get_feature_importance()

            for feature in trainer_importance:
                assert feature in predictor_importance
                assert abs(trainer_importance[feature] - predictor_importance[feature]) < 1e-6


class TestDataPipeline:
    """Integration tests for the data pipeline."""

    def test_data_loading_and_preprocessing(self, sample_dataframe: pd.DataFrame) -> None:
        """Test data loading and preprocessing integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            sample_dataframe.to_csv(csv_path, index=False)

            df = load_data(csv_path)

            preprocessor = DataPreprocessor()

            for _, row in df.iterrows():
                input_data = row.drop("Stress_Level").to_dict()

                warnings = preprocessor.validate_input(input_data)
                encoded = preprocessor.encode_categorical(input_data)
                features = preprocessor.transform_single(input_data)

                assert features.shape == (1, 10)

    def test_batch_processing(self, sample_dataframe: pd.DataFrame) -> None:
        """Test batch processing of multiple records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"

            trainer = ModelTrainer()
            X_train, X_test, y_train, y_test = trainer.prepare_data(sample_dataframe)
            trainer.train(X_train, y_train)
            trainer.evaluate(X_train, X_test, y_train, y_test)
            trainer.save_model(model_path)

            predictor = StressPredictor(model_path=model_path)
            predictor.load()

            batch_inputs = [
                {
                    "Hours_Worked": int(row["Hours_Worked"]),
                    "Sleep_Hours": int(row["Sleep_Hours"]),
                    "Bugs": int(row["Bugs"]),
                    "Deadline_Days": int(row["Deadline_Days"]),
                    "Coffee_Cups": int(row["Coffee_Cups"]),
                    "Meetings": int(row["Meetings"]),
                    "Interruptions": int(row["Interruptions"]),
                    "Experience_Years": row["Experience_Years"],
                    "Code_Complexity": row["Code_Complexity"],
                    "Remote_Work": row["Remote_Work"],
                }
                for _, row in sample_dataframe.head(10).iterrows()
            ]

            results = predictor.predict_batch(batch_inputs)

            assert len(results) == 10
            assert all(r.get("success", True) for r in results)
            assert all(0 <= r["stress_level"] <= 100 for r in results)
