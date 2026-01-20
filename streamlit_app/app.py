"""Main Streamlit application for Developer Stress Prediction."""

import sys
from pathlib import Path

# Add streamlit_app to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from components import (
    render_monitoring_dashboard,
    render_prediction_form,
    render_results,
)
from utils import check_api_health, predict_stress

st.set_page_config(
    page_title="Developer Stress Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    """Main application entry point."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Stress Prediction", "Monitoring Dashboard", "About"],
        index=0,
    )

    api_status = check_api_health()
    if api_status:
        st.sidebar.success("API Status: Connected")
    else:
        st.sidebar.error("API Status: Disconnected")

    st.sidebar.divider()
    st.sidebar.markdown(
        """
        **Developer Stress Predictor**

        ML-powered tool to predict and manage
        developer stress levels.

        Built with:
        - FastAPI
        - Streamlit
        - scikit-learn
        """
    )

    if page == "Stress Prediction":
        render_prediction_page()
    elif page == "Monitoring Dashboard":
        render_monitoring_dashboard()
    else:
        render_about_page()


def render_prediction_page() -> None:
    """Render the main prediction page."""
    st.markdown('<p class="main-header">Developer Stress Predictor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Predict stress levels based on work patterns and habits</p>',
        unsafe_allow_html=True,
    )

    if not check_api_health():
        st.error(
            """
            **API Not Available**

            The prediction API is not responding. Please ensure:
            1. The API server is running (`uvicorn src.api.main:app`)
            2. The API URL is correctly configured
            3. The model file exists in the `models/` directory
            """
        )
        return

    input_data = render_prediction_form()

    if input_data:
        with st.spinner("Analyzing stress factors..."):
            try:
                prediction = predict_stress(input_data)
                render_results(prediction)

                if "history" not in st.session_state:
                    st.session_state.history = []

                st.session_state.history.append(
                    {
                        "input": input_data,
                        "prediction": prediction["stress_level"],
                    }
                )

                if len(st.session_state.history) > 10:
                    st.session_state.history = st.session_state.history[-10:]

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

    if "history" in st.session_state and st.session_state.history:
        with st.expander("Prediction History"):
            for i, entry in enumerate(reversed(st.session_state.history)):
                st.write(f"**Prediction {len(st.session_state.history) - i}:** {entry['prediction']:.1f}")


def render_about_page() -> None:
    """Render the about page."""
    st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)

    st.markdown(
        """
        ## Developer Stress Prediction

        This application uses machine learning to predict developer stress levels
        based on various work-related factors.

        ### Features Used

        | Feature | Description |
        |---------|-------------|
        | Hours Worked | Average hours worked per day |
        | Sleep Hours | Average hours of sleep per night |
        | Bugs | Current number of bugs to fix |
        | Deadline Days | Days remaining until deadline |
        | Coffee Cups | Daily coffee consumption |
        | Meetings | Number of daily meetings |
        | Interruptions | Daily interruptions count |
        | Experience | Developer experience level |
        | Code Complexity | Complexity of current project |
        | Remote Work | Remote work status |

        ### Model Information

        - **Algorithm:** Random Forest Regressor
        - **Training Data:** 500 developer records
        - **Performance:** R² = 0.89 on test set

        ### Key Findings

        1. **Sleep hours** has the highest impact on stress
        2. **Hours worked** and **interruptions** are significant factors
        3. **Remote work** status has minimal impact

        ### Usage

        1. Navigate to "Stress Prediction" page
        2. Enter your work-related information
        3. Click "Predict Stress Level"
        4. Review results and recommendations

        ---

        **Version:** 1.0.0

        **Built with:** FastAPI, Streamlit, scikit-learn
        """
    )


if __name__ == "__main__":
    main()
