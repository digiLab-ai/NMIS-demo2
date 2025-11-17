# Cryogenic Isotope Separation — GP Emulator Demo

This is a minimal, physics-inspired demo that generates synthetic data for a cryogenic H/D/T
distillation segment. It maps operating/geometry inputs + feed moles to product mole shifts.

Now includes:
- Dataset description & variable glossary (expander)
- CSV downloads under the tables
- Output visualisation (pie charts for top/bottom splits)
- Validation with **uploaded** CSVs (ground truth, prediction, uncertainty) and metrics

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
