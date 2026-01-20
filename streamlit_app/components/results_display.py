"""Results display component."""

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from utils import get_stress_color, get_stress_description


def render_results(prediction: dict[str, Any]) -> None:
    """Render prediction results with visualizations.

    Args:
        prediction: Prediction result from API.
    """
    stress_level = prediction["stress_level"]
    warnings = prediction.get("warnings", [])

    st.subheader("Prediction Results")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=stress_level,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Stress Level", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": get_stress_color(stress_level)},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, 20], "color": "#E8F8F5"},
                        {"range": [20, 40], "color": "#D5F5E3"},
                        {"range": [40, 60], "color": "#FCF3CF"},
                        {"range": [60, 80], "color": "#FDEBD0"},
                        {"range": [80, 100], "color": "#FADBD8"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 80,
                    },
                },
            )
        )

        fig.update_layout(
            height=300,
            margin={"t": 50, "b": 0, "l": 30, "r": 30},
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric(
            label="Stress Level",
            value=f"{stress_level:.1f}",
            delta=None,
        )

        description = get_stress_description(stress_level)
        color = get_stress_color(stress_level)

        st.markdown(
            f"""
            <div style="
                padding: 1rem;
                border-radius: 0.5rem;
                background-color: {color}20;
                border-left: 4px solid {color};
            ">
                <strong>Assessment:</strong><br>
                {description}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if warnings:
        st.warning("**Warnings:**")
        for warning in warnings:
            st.write(f"- {warning}")

    st.divider()

    render_recommendations(stress_level)


def render_recommendations(stress_level: float) -> None:
    """Render recommendations based on stress level.

    Args:
        stress_level: Predicted stress level.
    """
    st.subheader("Recommendations")

    if stress_level >= 80:
        recommendations = [
            "Take immediate breaks - consider a day off if possible",
            "Talk to your manager about workload",
            "Prioritize sleep - aim for at least 7 hours",
            "Reduce meeting time where possible",
            "Consider pair programming to share the load",
        ]
        st.error("High stress detected. Consider the following immediate actions:")
    elif stress_level >= 60:
        recommendations = [
            "Schedule regular short breaks (Pomodoro technique)",
            "Review and prioritize your task list",
            "Limit after-hours work",
            "Reduce coffee intake if over 4 cups/day",
            "Communicate blockers early to the team",
        ]
        st.warning("Elevated stress levels. Consider these preventive measures:")
    elif stress_level >= 40:
        recommendations = [
            "Maintain your current work-life balance",
            "Keep communication channels open with your team",
            "Continue regular exercise routine",
            "Stay hydrated and take lunch breaks",
        ]
        st.info("Moderate stress. Keep up the good practices:")
    else:
        recommendations = [
            "Great job maintaining low stress levels!",
            "Continue your current habits",
            "Consider mentoring others with stress management",
            "Document what works well for you",
        ]
        st.success("Low stress levels. You're doing great!")

    for rec in recommendations:
        st.write(f"- {rec}")
