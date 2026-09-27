## HealthSense: Disease Forecasting 🧠🩺

**HealthSense** is an AI‑powered web application that predicts multiple diseases based on patient data and symptoms. The system integrates classical Machine Learning, Deep Learning, and a reinforcement learning prototype, delivering fast, accurate forecasts and interactive visualizations via a Streamlit interface.

---

## 🚀 Key Features

- **Multi‑Disease Support**: Diabetes, Heart Disease, Lung Cancer, Kidney Disease, Hypertension, Breast Cancer and more.
- **Machine Learning Models**: Decision Tree, Random Forest, SVM, KNN, Logistic Regression for each disease.
- **Deep Learning Models**: Feedforward ANN for tabular data; CNN for imaging data (e.g., lung scans).
- **Reinforcement Learning Prototype**: DQN agent simulating adaptive treatment strategies.
- **Interactive Visualizations**: Probability bars, correlation heatmaps, feature distributions, and Plotly charts.
- **User‑Friendly Interface**: Streamlit-based web app for symptom input, live predictions, and result exploration.
- **Modular Codebase**: Clean separation of data, models, notebooks, and application code.

---

## 📦 Project Structure

```
HealthSense-Disease-Forecasting/
├── app/
│   └── app.py                 # Streamlit application
├── data/
│   ├── raw/                   # Downloaded Kaggle CSV files
│   │   ├── diabetes.csv
│   │   ├── heart.csv
│   │   ├── lung_cancer.csv
│   │   ├── kidney.csv
│   │   ├── hypertension.csv
│   │   └── breast_cancer.csv
│   └── processed/             # Cleaned and feature‑engineered data
├── models/
│   ├── ml/                    # Serialized ML models
│   │   ├── diabetes_rf.pkl
│   │   ├── heart_rf.pkl
│   │   ├── lung_cancer_rf.pkl
│   │   ├── kidney_rf.pkl
│   │   ├── hypertension_rf.pkl
│   │   └── breast_cancer_rf.pkl
│   ├── dl/                    # Saved DL models
│   │   ├── diabetes_ann.h5
│   │   └── lung_cnn.h5
│   └── rl/                    # RL agent (DQN) checkpoints
│       └── treatment_agent.zip
├── notebooks/
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   ├── ML_Training.ipynb      # Training and evaluation of ML models
│   ├── DL_Training.ipynb      # Building and tuning DL models
│   └── RL_Experiments.ipynb   # Reinforcement Learning prototype
├── src/
│   ├── preprocess.py          # Data cleaning & feature engineering
│   ├── train_{disease}.py     # Scripts for ML training per disease (e.g., train_diabetes.py)
│   ├── train_dl.py            # DL training script
│   ├── train_rl.py            # RL training script
│   └── visualizer.py          # Functions to generate charts and plots
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview and instructions
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/HassanAnees1/HealthSense-Disease-Forecasting.git
   cd HealthSense-Disease-Forecasting
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Unix/macOS
   venv\\Scripts\\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Kaggle datasets** into `data/raw/`:
   - diabetes.csv
   - heart.csv
   - lung_cancer.csv
   - kidney.csv
   - hypertension.csv
   - breast_cancer.csv

   ### Dataset Links
   -[Diabetes (Pima Indians Diabetes Database)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database )

   -[Heart Disease (UCI Heart Disease Data)](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

   -[Lung Cancer](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer )

   -[Chronic Kidney Disease](https://www.kaggle.com/datasets/mansoordaku/ckdisease)

   -[Hypertension Risk Model](https://www.kaggle.com/datasets/khan1803115/hypertension-risk-model-main)

   -[Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) 

5. **Process data & train models**
   ```bash
   python src/preprocess.py
   python src/train_diabetes.py
   python src/train_heart.py
   python src/train_lung_cancer.py
   python src/train_kidney.py
   python src/train_hypertension.py
   python src/train_breast_cancer.py
   python src/train_dl.py
   python src/train_rl.py
   ```

6. **Run the Streamlit application**
   ```bash
   streamlit run app/app.py
   ```

---

## 👨‍💻 Team & Contacts

- **Hassan Anees** – Project Coordinator, DL & RL, Visualization  
  LinkedIn: https://www.linkedin.com/in/hassananees
- **Adham  ُElsayed** – Data Engineering & ML Training  
  LinkedIn: https://www.linkedin.com/in/adham_profile
- **Kirols ** – Backend & Frontend Integration, Streamlit UI  
  LinkedIn: https://www.linkedin.com/in/kirols_profile

Feel free to connect with us on LinkedIn for any questions or collaboration! 🌐

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

**Made with ❤️ by Team HealthSense**

