🔐 Network Intrusion Detection with Machine Learning (UNSW-NB15)
This project implements a network intrusion detection system (NIDS) using supervised machine learning on the UNSW-NB15 dataset. It supports both binary classification (normal vs attack) and multiclass classification (differentiating among specific attack types), providing clear performance metrics and visualizations for model evaluation.


📦 Dataset
Source: UNSW-NB15 Dataset

Description: Realistic modern network traffic captured using IXIA PerfectStorm and labeled with various attack types.

Classes: Includes attacks such as Exploits, DoS, Fuzzers, Reconnaissance, Shellcode, Backdoor, Worms, and Generic.


🧠 Features
🔎 Binary Classification: Detect whether a connection is normal or malicious.

🧩 Multiclass Classification: Identify the specific attack type out of 10+ classes.

⚖️ Balanced Sampling: Ensures fair evaluation by sampling equal data per class.

📊 Metrics & Visualization:
- Classification reports (precision, recall, F1-score)
- Confusion matrices (Seaborn heatmaps)
- Feature importance charts
- Per-class data distribution reporting


⚙️ Machine Learning Stack
- pandas and numpy for data wrangling
- scikit-learn for preprocessing, training, and evaluation
- matplotlib and seaborn for data visualization


📈 Models Used
- Random Forest Classifier
- Chosen for its robustness with imbalanced and high-dimensional data.
- Trained separately for binary and multiclass settings.


📂 Project Structure

unsw_ids_model.py         # Main script for loading, preprocessing, training, and evaluation
unsw_nb15_labeled.csv     # Cleaned and labeled dataset (not committed due to size)
README.md                 # You're here!


🚀 Getting Started

Install dependencies:
pip install -r requirements.txt

Run the model:
python unsw_ids_model.py


📌 Results Snapshot

Example binary classification metrics:
Accuracy: 99.3%
Precision, Recall, F1: near-perfect on balanced classes

Example confusion matrix for multiclass:
High accuracy on Exploits, DoS, Normal
Room for improvement on low-frequency classes like Worms or Shellcode


💡 Future Improvements
- Add additional classifiers (XGBoost, SVM)
- Explore time-series or sequence modeling
- Perform hyperparameter tuning and feature selection
- Deploy as an interactive dashboard or API


📜 License
This project is for academic and educational purposes. Attribution for dataset goes to UNSW Canberra.
