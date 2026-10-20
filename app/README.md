# App

Streamlit interface for the frozen resolution pipeline.

From the project folder:

```bash
streamlit run app/app.py
```

The process loads `models/frozen_resolution_pipeline.joblib` through a path relative to the repository. It does not fit a model, run a notebook, or read the raw extract.

On Render, the start command is:

```bash
streamlit run app/app.py --server.address 0.0.0.0 --server.port $PORT
```
