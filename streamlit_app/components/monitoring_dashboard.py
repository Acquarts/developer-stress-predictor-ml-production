"""Monitoring dashboard component."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import check_api_health, get_feature_importance, get_model_info


def render_monitoring_dashboard() -> None:
    """Render the monitoring dashboard."""
    st.subheader("System Monitoring")

    col1, col2, col3 = st.columns(3)

    api_healthy = check_api_health()

    with col1:
        st.metric(
            label="API Status",
            value="Healthy" if api_healthy else "Unhealthy",
            delta=None,
        )

    if not api_healthy:
        st.error("API is not responding. Please check if the service is running.")
        return

    try:
        model_info = get_model_info()

        with col2:
            st.metric(
                label="Model R² Score",
                value=f"{model_info['metrics'].get('test_r2', 0):.3f}",
            )

        with col3:
            st.metric(
                label="Estimators",
                value=model_info["n_estimators"],
            )

    except Exception as e:
        st.error(f"Error fetching model info: {e}")
        return

    st.divider()

    render_feature_importance()

    st.divider()

    render_model_details(model_info)


def render_feature_importance() -> None:
    """Render feature importance chart."""
    st.subheader("Feature Importance")

    try:
        importance = get_feature_importance()

        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        feature_labels = {
            "Hours_Worked": "Hours Worked",
            "Sleep_Hours": "Sleep Hours",
            "Bugs": "Bugs",
            "Deadline_Days": "Deadline Days",
            "Coffee_Cups": "Coffee Cups",
            "Meetings": "Meetings",
            "Interruptions": "Interruptions",
            "Experience_Years": "Experience",
            "Code_Complexity": "Code Complexity",
            "Remote_Work": "Remote Work",
        }

        labels = [feature_labels.get(k, k) for k in sorted_importance.keys()]
        values = list(sorted_importance.values())

        fig = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color="#3498DB",
            )
        )

        fig.update_layout(
            title="Feature Importance in Stress Prediction",
            xaxis_title="Importance Score",
            yaxis_title="",
            height=400,
            margin={"t": 50, "b": 50, "l": 150, "r": 30},
            yaxis={"categoryorder": "total ascending"},
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            """
            **Interpretation:**
            - **Sleep Hours** has the highest impact on stress prediction
            - **Hours Worked** and **Interruptions** are also significant factors
            - Remote work status has minimal impact on stress levels
            """
        )

    except Exception as e:
        st.error(f"Error loading feature importance: {e}")


def render_model_details(model_info: dict) -> None:
    """Render detailed model information.

    Args:
        model_info: Model information dictionary.
    """
    st.subheader("Model Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Configuration**")
        st.json(
            {
                "Model Type": model_info["model_type"],
                "Number of Estimators": model_info["n_estimators"],
                "Max Depth": model_info["max_depth"],
            }
        )

    with col2:
        st.markdown("**Training Metrics**")
        metrics = model_info.get("metrics", {})
        if metrics:
            metrics_display = {
                "Train R²": f"{metrics.get('train_r2', 0):.4f}",
                "Test R²": f"{metrics.get('test_r2', 0):.4f}",
                "Train RMSE": f"{metrics.get('train_rmse', 0):.2f}",
                "Test RMSE": f"{metrics.get('test_rmse', 0):.2f}",
            }
            st.json(metrics_display)

    st.markdown("**Feature Columns**")
    st.write(", ".join(model_info.get("feature_columns", [])))
