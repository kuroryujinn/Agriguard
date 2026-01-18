import streamlit as st
import os
import sys
import subprocess
import json
import tempfile
import pandas as pd
import time
from pathlib import Path

# 1) Define APP_DIR as the directory containing web_app.py
APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Agriguard - Tree Survival Analysis", layout="wide")

st.title("🌳 Agriguard: Tree Survival Analysis")
st.markdown("""
Upload high-resolution drone orthomosaics from three different stages (Pitting, Planting, Sowing) 
to analyze sapling survival rates using CV and Gemini AI.
""")

# Sidebar for configuration
st.sidebar.header("Pipeline Configuration")

mode_options = ["hybrid", "cv"]
gemini_key = os.getenv("GEMINI_API_KEY")
default_mode = "hybrid" if gemini_key else "cv"

mode = st.sidebar.selectbox("Analysis Mode", mode_options, index=mode_options.index(default_mode))
if mode == "hybrid" and not gemini_key:
    st.sidebar.warning("GEMINI_API_KEY not found. Will downgrade to CV mode.")

max_gemini_calls = st.sidebar.number_input("Max Gemini Calls", value=50, min_value=1)
crop_size = st.sidebar.number_input("Crop Size (px)", value=256, min_value=64, step=32)
offset_px = st.sidebar.number_input("Local Search Offset (px)", value=40, min_value=0, step=5)

st.sidebar.subheader("Thresholds")
alive_threshold = st.sidebar.slider("Alive Threshold", 0.0, 0.2, 0.05, 0.01)
dead_threshold = st.sidebar.slider("Dead Threshold", 0.0, 0.1, 0.01, 0.005)
planted_threshold = st.sidebar.slider("Planted Threshold", 0.0, 0.1, 0.02, 0.005)

# Main Content: Uploaders
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Post Pitting (OP1)")
    op1_file = st.file_uploader("Upload OP1", type=["tif", "tiff", "jpg", "jpeg", "png"], key="op1")

with col2:
    st.subheader("2. Post Planting (OP2)")
    op2_file = st.file_uploader("Upload OP2", type=["tif", "tiff", "jpg", "jpeg", "png"], key="op2")

with col3:
    st.subheader("3. Post Sowing (OP3)")
    op3_file = st.file_uploader("Upload OP3", type=["tif", "tiff", "jpg", "jpeg", "png"], key="op3")

# Run Analysis
ready = op1_file and op2_file and op3_file
run_button = st.button("🚀 Run Analysis", disabled=not ready, use_container_width=True)

if run_button:
    with st.status("Running Pipeline...", expanded=True) as status:
        # 1. Save to temp files
        st.write("Preparing images...")
        with tempfile.TemporaryDirectory() as temp_dir:
            op1_path = os.path.join(temp_dir, op1_file.name)
            op2_path = os.path.join(temp_dir, op2_file.name)
            op3_path = os.path.join(temp_dir, op3_file.name)

            with open(op1_path, "wb") as f: f.write(op1_file.getbuffer())
            with open(op2_path, "wb") as f: f.write(op2_file.getbuffer())
            with open(op3_path, "wb") as f: f.write(op3_file.getbuffer())

            # 2. Build command
            cmd = [
                sys.executable, "user.py",
                "--op1", op1_path,
                "--op2", op2_path,
                "--op3", op3_path,
                "--mode", mode,
                "--max-gemini-calls", str(max_gemini_calls),
                "--crop-size", str(crop_size),
                "--offset-px", str(offset_px),
                "--alive-threshold", str(alive_threshold),
                "--dead-threshold", str(dead_threshold),
                "--planted-threshold", str(planted_threshold)
            ]

            st.write("Executing user.py logic...")
            # 2) Run user.py using subprocess with cwd=APP_DIR
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(APP_DIR)
            )

            # 3. Stream output
            output_container = st.empty()
            full_output = ""
            for line in process.stdout:
                full_output += line
                output_container.code(full_output, language="text")
            
            process.wait()

            # 3b) Debug output: list files in user_outputs
            out_dir = APP_DIR / "user_outputs"
            st.info(f"Debug: user_outputs path: {out_dir.absolute()}")
            if out_dir.exists():
                st.info(f"Debug: Files in user_outputs: {os.listdir(out_dir)}")
            else:
                st.warning("Debug: user_outputs directory not found")

            if process.returncode == 0:
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                st.success("Analysis finished successfully.")
            else:
                status.update(label="Analysis Failed", state="error", expanded=True)
                st.error(f"Pipeline exited with error code {process.returncode}")
                # 4b) Display stderr (it's already in full_output as we used stderr=STDOUT)
                st.code(full_output, language="text")
                st.stop()

        # 4. Display Results
        st.divider()
        st.header("📊 Results Summary")
        
        # 3) Read outputs from APP_DIR / "user_outputs" (absolute path)
        out_dir = APP_DIR / "user_outputs"
        summary_path = out_dir / "summary.json"
        
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summary = json.load(f)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Pits Found", summary.get("total_pits", 0))
            m2.metric("Total Planted", summary.get("total_planted", 0))
            m3.metric("Alive (Survival)", summary.get("alive", 0))
            m4.metric("Dead", summary.get("dead", 0))

            m5, m6, m7 = st.columns(3)
            m5.metric("Survival Rate", f"{summary.get('survival_pct', 0)}%")
            m6.metric("Gemini Calls", summary.get("gemini_calls_used", 0))
            m7.metric("Runtime", f"{summary.get('runtime_seconds', 0)}s")
        else:
            # 4a) Show a clear error in the UI if summary.json missing
            st.error(f"Critical Error: Summary results not found at {summary_path}")
            st.warning("Check the pipeline output above for errors.")

        # 5. Downloads
        st.divider()
        st.header("📥 Download Reports")
        d_col1, d_col2 = st.columns(2)

        surv_path = out_dir / "op3_survival.csv"
        if surv_path.exists():
            df_s = pd.read_csv(surv_path)
            d_col1.download_button(
                label="Download Full Survival CSV",
                data=df_s.to_csv(index=False),
                file_name="op3_survival.csv",
                mime="text/csv",
                use_container_width=True
            )

        dead_path = out_dir / "dead_locations.csv"
        if dead_path.exists():
            df_d = pd.read_csv(dead_path)
            d_col2.download_button(
                label="Download Dead Locations CSV",
                data=df_d.to_csv(index=False),
                file_name="dead_locations.csv",
                mime="text/csv",
                use_container_width=True
            )

elif not ready:
    st.info("Please upload all three images to start the analysis.")
