"""Input checks for the frozen-pipeline application.

The valid-input test scores the exported pipeline. It does not fit a model.
The demonstration sample is a set of frequent training categories, not a
municipal case.
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.predict import (  # noqa: E402
    ACADEMIC_NOTICE,
    SERVICE_WARNING,
    SUPPORTED_DATE_END,
    SUPPORTED_DATE_START,
    InputError,
    load_pipeline,
    parse_created_date,
    pipeline_path,
    predict_request,
    validate_request,
)
from src.config import TARGET_LABELS  # noqa: E402
from src.preprocessing import MODEL_FEATURES  # noqa: E402


def _pipeline():
    path = pipeline_path()
    if not path.is_file():
        raise unittest.SkipTest(f"Exported pipeline is not at {path}.")
    return load_pipeline(path)


class PipelineLoadTest(unittest.TestCase):
    def test_missing_file_tells_the_user_not_to_expect_training(self) -> None:
        with self.assertRaises(FileNotFoundError) as caught:
            load_pipeline(PROJECT_ROOT / "models" / "does_not_exist.joblib")
        self.assertIn("does not train", str(caught.exception))

    def test_saved_features_match_training(self) -> None:
        pipeline = _pipeline()
        self.assertEqual(pipeline["features"], list(MODEL_FEATURES))
        self.assertNotIn("Work_Center", pipeline["features"])
        self.assertFalse(pipeline["model"].early_stopping)
        self.assertIn("not an unresolved case", pipeline["sample_input"]["description"])
        self.assertNotIn("Notification", pipeline["sample_input"])


class ValidInputTest(unittest.TestCase):
    def test_demonstration_sample_returns_probabilities(self) -> None:
        pipeline = _pipeline()
        sample = {
            key: value
            for key, value in pipeline["sample_input"].items()
            if key != "description"
        }
        result = predict_request(pipeline, sample)
        self.assertIn(result["prediction"], TARGET_LABELS)
        self.assertEqual(set(result["probabilities"]), set(TARGET_LABELS))
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)
        self.assertEqual(result["unknown_categories"], [])
        self.assertEqual(list(result["model_inputs"]), list(MODEL_FEATURES))
        self.assertNotIn("Work_Center", result["model_inputs"])
        self.assertEqual(result["academic_notice"], ACADEMIC_NOTICE)
        self.assertEqual(result["service_warning"], SERVICE_WARNING)
        self.assertIn("project label", result["meaning"])
        highest = max(result["probabilities"], key=result["probabilities"].get)
        self.assertEqual(result["prediction"], highest)


class MissingInputTest(unittest.TestCase):
    def setUp(self) -> None:
        pipeline = _pipeline()
        self.pipeline = pipeline
        self.sample = {
            key: value
            for key, value in pipeline["sample_input"].items()
            if key != "description"
        }

    def test_missing_complaint_type(self) -> None:
        payload = dict(self.sample)
        del payload["C3_Complaint_Type"]
        with self.assertRaises(InputError) as caught:
            validate_request(payload)
        self.assertIn("complaint type", str(caught.exception).lower())

    def test_blank_suburb(self) -> None:
        payload = dict(self.sample)
        payload["Suburb"] = "   "
        with self.assertRaises(InputError) as caught:
            validate_request(payload)
        self.assertIn("suburb", str(caught.exception).lower())

    def test_missing_date(self) -> None:
        payload = dict(self.sample)
        payload["Created_On_Date"] = None
        with self.assertRaises(InputError) as caught:
            validate_request(payload)
        self.assertIn("date", str(caught.exception).lower())

    def test_work_centre_is_rejected(self) -> None:
        payload = dict(self.sample)
        payload["Work_Center"] = "A depot"
        with self.assertRaises(InputError) as caught:
            predict_request(self.pipeline, payload)
        self.assertIn("Work_Center", str(caught.exception))


class UnknownCategoryTest(unittest.TestCase):
    def test_unknown_suburb_is_grouped_and_still_scored(self) -> None:
        pipeline = _pipeline()
        payload = {
            key: value
            for key, value in pipeline["sample_input"].items()
            if key != "description"
        }
        payload["Suburb"] = "ZZZ Not A Cape Town Suburb"
        result = predict_request(pipeline, payload)
        self.assertIn(result["prediction"], TARGET_LABELS)
        self.assertEqual(result["unknown_categories"], ["Suburb"])
        self.assertTrue(any("rare or unseen" in note for note in result["notes"]))
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)


class DateValidationTest(unittest.TestCase):
    def test_valid_training_date(self) -> None:
        self.assertEqual(parse_created_date("2024-06-12"), date(2024, 6, 12))
        self.assertEqual(parse_created_date(date(2024, 6, 12)), date(2024, 6, 12))

    def test_invalid_text(self) -> None:
        with self.assertRaises(InputError) as caught:
            parse_created_date("not-a-date")
        self.assertIn("not a valid calendar date", str(caught.exception))

    def test_impossible_calendar_day(self) -> None:
        with self.assertRaises(InputError) as caught:
            parse_created_date("2024-02-30")
        self.assertIn("not a valid calendar date", str(caught.exception))

    def test_before_extract(self) -> None:
        with self.assertRaises(InputError) as caught:
            parse_created_date(SUPPORTED_DATE_START.replace(year=2022))
        self.assertIn("1 January 2023", str(caught.exception))

    def test_after_extract(self) -> None:
        with self.assertRaises(InputError) as caught:
            parse_created_date(date(2026, 10, 7))
        self.assertIn("6 October 2026", str(caught.exception))
        self.assertEqual(SUPPORTED_DATE_END, date(2026, 10, 6))

    def test_date_after_training_fit_adds_a_note(self) -> None:
        pipeline = _pipeline()
        payload = {
            key: value
            for key, value in pipeline["sample_input"].items()
            if key != "description"
        }
        payload["Created_On_Date"] = "2026-03-15"
        result = predict_request(pipeline, payload)
        self.assertTrue(any("31 December 2024" in note for note in result["notes"]))
        self.assertIn(result["prediction"], TARGET_LABELS)


class StreamlitFlowTest(unittest.TestCase):
    def _app(self):
        if not pipeline_path().is_file():
            self.skipTest(f"Exported pipeline is not at {pipeline_path()}.")
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=90)
        app.run()
        if app.exception:
            raise app.exception[0].value
        return app

    def _joined_text(self, app) -> str:
        parts = []
        for collection in (
            app.markdown,
            app.error,
            app.warning,
            app.info,
            app.caption,
            app.title,
            app.subheader,
        ):
            for element in collection:
                parts.append(str(getattr(element, "value", "")))
        return "\n".join(parts)

    def test_page_shows_notices_and_rejects_an_empty_form(self) -> None:
        app = self._app()
        text = self._joined_text(app).lower()
        self.assertIn("academic prototype", text)
        self.assertIn("not an official city of cape town system", text)
        self.assertIn("deny or reduce service", text)
        self.assertIn("work centre is not collected", text)
        labels = [select.label for select in app.selectbox]
        self.assertEqual(labels, ["Complaint type", "Suburb", "Ward", "Subcouncil"])
        estimate = next(button for button in app.button if "Estimate" in button.label)
        estimate.click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("complaint type" in str(error.value).lower() for error in app.error))

    def test_demonstration_sample_produces_an_outcome(self) -> None:
        app = self._app()
        app.button(key="fill_sample").click().run()
        self.assertFalse(app.exception)
        estimate = next(button for button in app.button if "Estimate" in button.label)
        estimate.click().run()
        self.assertFalse(app.exception)
        text = self._joined_text(app)
        self.assertTrue(any(label in text for label in TARGET_LABELS))
        self.assertIn("does not establish", text)
        self.assertIn("model confidence", text.lower())
        self.assertNotIn("Notification", text)


class InterfaceCopyTest(unittest.TestCase):
    def test_app_does_not_fit_and_states_the_notices(self) -> None:
        source = (PROJECT_ROOT / "app" / "app.py").read_text(encoding="utf-8")
        self.assertNotIn(".fit(", source)
        self.assertNotIn("fit_frozen", source)
        self.assertNotIn("export_frozen", source)
        self.assertIn("ACADEMIC_NOTICE", source)
        self.assertIn("SERVICE_WARNING", source)
        self.assertIn("WORK_CENTRE_NOTICE", source)
        form = source.split("with st.form")[1]
        self.assertNotIn("Work", form)
        self.assertIn("FIELD_LABELS[column]", source)
        self.assertIn("Fill demonstration sample", source)


if __name__ == "__main__":
    unittest.main()
