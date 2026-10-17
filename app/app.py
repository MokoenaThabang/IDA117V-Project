"""Streamlit form for the frozen Cape Town resolution classifier.

Launch from the project folder:

    streamlit run app/app.py

The process loads models/frozen_resolution_pipeline.joblib. It does not fit
a model and it does not rebuild the encoder.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.predict import (
    ACADEMIC_NOTICE,
    FIELD_LABELS,
    SERVICE_CREATED_DATE_COLUMN,
    SERVICE_WARNING,
    SOURCE_CATEGORICAL_COLUMNS,
    SUPPORTED_DATE_END,
    SUPPORTED_DATE_START,
    WORK_CENTRE_NOTICE,
    InputError,
    load_pipeline,
    predict_request,
)
from src.config import TARGET_LABELS

st.set_page_config(
    page_title="Cape Town service-request resolution classifier",
    layout="centered",
)


def _widget_default(key: str, value):
    """Pass a default only before session state exists for this widget."""
    if key in st.session_state:
        return {}
    return {"value": value} if key == SERVICE_CREATED_DATE_COLUMN else {"index": value}


def _apply_sample(pipeline: dict) -> None:
    sample = pipeline["sample_input"]
    for column in SOURCE_CATEGORICAL_COLUMNS:
        st.session_state[column] = sample[column]
    st.session_state[SERVICE_CREATED_DATE_COLUMN] = pd.Timestamp(sample["Created_On_Date"]).date()


def _show_result(result: dict) -> None:
    st.subheader("Estimated outcome")
    st.markdown(f"**{result['prediction']}**")
    st.write(result["meaning"])
    st.write(
        "This is the class with the model's highest predicted probability for the values below. "
        "It is not a record of a City outcome, and it does not establish what will cause the request "
        "to be completed or to stay open."
    )
    probability_table = pd.DataFrame(
        {
            "Outcome": list(TARGET_LABELS),
            "Predicted probability": [result["probabilities"][label] for label in TARGET_LABELS],
        }
    )
    st.dataframe(
        probability_table,
        hide_index=True,
        width="stretch",
        column_config={
            "Predicted probability": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        "These probabilities come from the fitted model. Training used balanced class weights, "
        "so the numbers are not the share of historical requests in each class. "
        "A high probability can still be wrong. Read it as model confidence for this input, "
        "not as a service standard and not as a reason the request will finish or remain open."
    )
    for note in result["notes"]:
        st.info(note)
    with st.expander("Values passed to the fitted pipeline"):
        displayed = []
        for name, value in result["model_inputs"].items():
            if name == "created_is_weekend":
                displayed.append("1" if float(value) >= 0.5 else "0")
            else:
                displayed.append(str(value))
        rows = pd.DataFrame({"Input": list(result["model_inputs"]), "Value": displayed})
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            "The creation date is used only to derive the weekday, the month, and the weekend flag. "
            "Work centre is not in this list."
        )


def main() -> None:
    st.title("Cape Town service-request resolution classifier")
    st.write(
        "Estimate, at the time a municipal service request is logged, whether it is likely "
        "to be resolved within seven days, resolved later, or still unresolved after the "
        "seven-day observation window."
    )
    st.error(ACADEMIC_NOTICE)
    st.warning(SERVICE_WARNING)

    try:
        pipeline = load_pipeline()
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

    st.info(WORK_CENTRE_NOTICE)
    st.caption(pipeline["sample_input"]["description"])
    if st.button("Fill demonstration sample", key="fill_sample"):
        _apply_sample(pipeline)
        st.rerun()

    with st.form("resolution_request"):
        entered = {}
        for column in SOURCE_CATEGORICAL_COLUMNS:
            entered[column] = st.selectbox(
                FIELD_LABELS[column],
                options=pipeline["category_options"][column],
                placeholder=f"Select a {FIELD_LABELS[column].lower()}",
                key=column,
                **_widget_default(column, None),
            )
        entered[SERVICE_CREATED_DATE_COLUMN] = st.date_input(
            FIELD_LABELS[SERVICE_CREATED_DATE_COLUMN],
            min_value=SUPPORTED_DATE_START,
            max_value=SUPPORTED_DATE_END,
            format="YYYY-MM-DD",
            key=SERVICE_CREATED_DATE_COLUMN,
            help="Used only to derive the weekday, the month, and whether the day is a weekend.",
            **_widget_default(SERVICE_CREATED_DATE_COLUMN, None),
        )
        submitted = st.form_submit_button("Estimate resolution outcome")

    if submitted:
        try:
            result = predict_request(pipeline, entered)
        except InputError as exc:
            st.error(str(exc))
        else:
            _show_result(result)


main()
