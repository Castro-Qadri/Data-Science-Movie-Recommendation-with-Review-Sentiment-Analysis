Project status — Movie Recommendation with Sentiment Analysis

Current state
- Notebook: `Recommendation_System.ipynb` contains the full pipeline (EDA, TF-IDF, recommendation, sentiment training, evaluation). New guide-aligned sections were added (5–9).
- Streamlit app: `WebApp.py` updated with cinematic theme, header, and footer (team details present).
- Artifacts: `Model/` contains `movies_data.joblib`, `similarity.joblib`, `sentiment_analysis_model.pkl`, and `tfidf_vectorizer.pkl` (if previously saved).

Completed
- EDA section with 10 plots (cells and image save paths present).
- Memory-safe recommendation using sparse TF-IDF + `linear_kernel`.
- Sentiment model training and comparison code in the notebook (requires IMDB dataset to run).
- README updated to reflect actual project, usage, and team.

Blockers / Missing
- `IMDB Dataset.csv` is missing from the workspace. Sections 7–9 (sentiment plots, model training & evaluation) need this file to run end-to-end.
- Notebook still contains legacy cells and intermediate outputs; a final cleaned notebook is not yet saved as the canonical submission (a skeleton was added).

Next recommended steps
1. Place `IMDB Dataset.csv` in `movie-project/` or `archive/` and run the notebook to generate the saved plot PNGs and trained models.
2. Decide whether you want me to remove the legacy cells from `Recommendation_System.ipynb` or to keep the full history; I can create a cleaned copy and keep the existing notebook as an archive.
3. Optionally I can commit and open a small final report (PDF/markdown) with the generated plots after you run the notebook locally and confirm outputs.

How to run (quick)
- Create & activate the venv (if not already):

```powershell
.\.venv311\Scripts\Activate.ps1
```

- Install dependencies:

```powershell
pip install -r movie-project\requirements.txt
```

- Run the Streamlit app:

```powershell
streamlit run movie-project\WebApp.py
```

- Run the notebook in your `ds_project_venv311` kernel and ensure `API_KEY` is set in your environment for poster fetching.

Notes
- I created `Recommendation_System_clean.ipynb` as a runnable skeleton you can execute to reproduce the core pipeline without legacy noise.
- Tell me whether to proceed with permanently cleaning `Recommendation_System.ipynb` (I will keep an archived copy before removing anything).