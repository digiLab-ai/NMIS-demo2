
import streamlit as st
from data import generate_dataset

st.set_page_config(page_title="Mechanical Jack Simulator", layout="wide")

st.title("Mechanical Jack Simulator (Synthetic Data)")

st.sidebar.header("Simulation Settings")

n_runs = st.sidebar.slider("Number of simulations", 10, 200, 50)
dt = st.sidebar.slider("Time step (s)", 0.5, 5.0, 1.0)

if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        df = generate_dataset(n_runs=n_runs, dt=dt)

    st.success("Simulation complete")

    st.subheader("Sample of Generated Data")
    st.dataframe(df.head())

    st.subheader("Lift Height vs Time (Run 0)")
    st.line_chart(df[df["run_id"] == 0].set_index("time_s")["height_m"])

    st.subheader("Required Torque vs Time (Run 0)")
    st.line_chart(df[df["run_id"] == 0].set_index("time_s")["torque_Nm"])
