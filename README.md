# Mouse metabolomics dashboard

Streamlit dashboard for exploring untargeted and targeted metabolomics data from mouse experiments.

The default workbook is `untargeted_PAA_2HB_GF_SPF_background.xlsx`, which includes:

- `Metadata`
- `NR_untargeted`
- `NR_PAA_2HB`

## Run locally

```bash
conda create -n metabolomics_dashboard python=3.11
conda activate metabolomics_dashboard
pip install -r requirements.txt
streamlit run app.py
```
