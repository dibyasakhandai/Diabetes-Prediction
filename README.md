# Diabetes Prediction — Streamlit App (Starter)

This project turns your SVM-based diabetes prediction into an interactive **Streamlit** web app.

## Folder contents
- `train_model.py` — trains an SVM, evaluates it, and saves `model.pkl`, `scaler.pkl`, and `feature_names.json`.
- `app.py` — Streamlit app that loads the saved artifacts and provides an interactive form.
- `requirements.txt` — Python dependencies to install.

> **You must provide** `diabetes.csv` in the same folder before running `train_model.py`.
Expected columns: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`.

---

## 1) Run locally (Windows/Mac/Linux)

1. **Install Python 3.9+** (if not already).
2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   # Activate
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Add your dataset**: Place `diabetes.csv` in this folder.
5. **Train and save artifacts:**
   ```bash
   python train_model.py
   ```
   You should see train/test accuracy and then files `model.pkl`, `scaler.pkl`, `feature_names.json` will be created.
6. **Launch the app:**
   ```bash
   streamlit run app.py
   ```
   Streamlit will open a browser window at `http://localhost:8501`.

---

## 2) Run in Google Colab

1. Open a new Colab notebook.
2. Upload the 4 project files (`train_model.py`, `app.py`, `requirements.txt`, `README.md`) and your `diabetes.csv`.
3. Install deps and train:
   ```python
   !pip -q install -r requirements.txt
   !python train_model.py
   ```
4. Expose the Streamlit app (two simple options):

   **Option A: LocalTunnel (no account needed)**
   ```python
   # Start Streamlit
   import subprocess, time
   streamlit_proc = subprocess.Popen(["streamlit", "run", "app.py", "--server.headless=true"])
   time.sleep(5)  # give it a moment to start

   # Install and run localtunnel to get a public URL
   !npm -g install localtunnel
   public_url = !lt --port 8501 --print-requests
   print(public_url)
   ```

   **Option B: Pyngrok (ngrok account needed)**
   ```python
   !pip -q install pyngrok
   from pyngrok import ngrok
   # ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # once per account
   public_url = ngrok.connect(8501)
   print(public_url)
   !streamlit run app.py --server.headless=true --server.port=8501
   ```

Open the printed URL to use your app.

---

## 3) Deploy on Streamlit Community Cloud

1. Push this folder to a **public GitHub repo** (include `app.py`, `requirements.txt`, `train_model.py`, and **the saved artifacts** `model.pkl`, `scaler.pkl`, `feature_names.json` — _do not_ commit `diabetes.csv` if it's private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub.
3. Select your repo and set **Main file** to `app.py`.
4. Add any **secrets** if needed (not required here).
5. Click **Deploy**.

---

## Notes & best practices

- Your original code had a small issue: you created `StandardScaler()` but called `transform()` without first fitting it. In `train_model.py` we do:
  - `scaler.fit_transform(X_train)` and **only** `scaler.transform(X_test)` to avoid data leakage.
- Keep feature order consistent between training and inference. We store it in `feature_names.json` and follow that order in the app.
- For nicer UX, consider adding:
  - `st.sidebar` help, validation ranges, and tooltips.
  - Charts showing distribution of inputs if you load the dataset inside the app (optional).
