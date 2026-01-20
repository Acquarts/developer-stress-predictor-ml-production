"""Prediction form component."""

from typing import Any

import streamlit as st


def render_prediction_form() -> dict[str, Any] | None:
    """Render the prediction input form.

    Returns:
        Dictionary with form values if submitted, None otherwise.
    """
    st.subheader("Enter Developer Information")

    col1, col2 = st.columns(2)

    with col1:
        hours_worked = st.slider(
            "Hours Worked per Day",
            min_value=4,
            max_value=15,
            value=8,
            help="Average hours worked per day",
        )

        sleep_hours = st.slider(
            "Hours of Sleep",
            min_value=3,
            max_value=8,
            value=7,
            help="Average hours of sleep per night",
        )

        bugs = st.number_input(
            "Number of Bugs",
            min_value=0,
            max_value=50,
            value=10,
            help="Current number of bugs to fix",
        )

        deadline_days = st.number_input(
            "Days Until Deadline",
            min_value=0,
            max_value=60,
            value=14,
            help="Days remaining until project deadline",
        )

        coffee_cups = st.slider(
            "Coffee Cups per Day",
            min_value=0,
            max_value=10,
            value=3,
            help="Average coffee consumption per day",
        )

    with col2:
        meetings = st.slider(
            "Number of Meetings",
            min_value=0,
            max_value=20,
            value=5,
            help="Average meetings per day",
        )

        interruptions = st.slider(
            "Daily Interruptions",
            min_value=0,
            max_value=10,
            value=4,
            help="Average interruptions per day",
        )

        experience = st.selectbox(
            "Experience Level",
            options=["Junior", "Mid", "Senior"],
            index=1,
            help="Developer experience level",
        )

        complexity = st.selectbox(
            "Code Complexity",
            options=["Low", "Medium", "High"],
            index=1,
            help="Complexity of current project code",
        )

        remote_work = st.selectbox(
            "Remote Work",
            options=["Yes", "No"],
            index=0,
            help="Working remotely?",
        )

    st.divider()

    if st.button("Predict Stress Level", type="primary", use_container_width=True):
        return {
            "Hours_Worked": hours_worked,
            "Sleep_Hours": sleep_hours,
            "Bugs": bugs,
            "Deadline_Days": deadline_days,
            "Coffee_Cups": coffee_cups,
            "Meetings": meetings,
            "Interruptions": interruptions,
            "Experience_Years": experience,
            "Code_Complexity": complexity,
            "Remote_Work": remote_work,
        }

    return None
