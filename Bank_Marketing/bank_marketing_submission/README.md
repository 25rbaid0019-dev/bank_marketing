# Bank Marketing Campaign Response Prediction Using Machine Learning

## Team
- E Ganesh Sai — 25RBAID0019
- Y Swaha Devi — 25RBAIM0078
- CSE (AI & ML), MNR University
- Academic Year: 2026–2027

## Project description
This project predicts whether a bank customer will subscribe to a term deposit (Yes/No) using a Decision Tree Classifier and the UCI Bank Marketing dataset.

## Dataset
**UCI Bank Marketing Dataset (dataset ID 222)**

Official source: https://archive.ics.uci.edu/dataset/222/bank+marketing

The project uses `bank-full.csv` (45,211 records, 17 columns). The dataset is not bundled here; the official UCI link above is provided as the dataset source.

## Algorithm / models used
- Decision Tree Classifier
- Class-balanced training
- Stratified 80:20 train/test split
- GridSearchCV with stratified 5-fold cross-validation
- One-hot encoding for categorical features
- Call `duration` excluded from model inputs to avoid post-call data leakage

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn and Scikit-learn.

## Install and run
1. Download `bank-full.csv` from the UCI link above and place it in this repository folder.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run:
   `python bank_marketing_decision_tree.py`
4. Results are written to `outputs/`.

## Results
The submitted report records the following test-set results for the tuned Decision Tree:

| Metric | Score |
|---|---:|
| Accuracy | 82.2% |
| Precision (Yes) | 34.4% |
| Recall (Yes) | 58.0% |
| F1-score | 0.432 |
| ROC-AUC | 0.775 |

The model was evaluated on 9,043 test customers. The report explains that `duration` is excluded because it is only known after the call and would cause data leakage.

## Repository contents
- `bank_marketing_decision_tree.py` — complete runnable source code
- `requirements.txt` — Python dependencies
- `Project_Report.docx` — project report
- `Project_Presentation.pptx` — presentation slides
- `Bank_Marketing_Mini_Project_Report.docx` — report copy
- `Bank_Marketing_Demo.mp4` — demo walkthrough video
- `data/DATASET_LINK.txt` — official dataset link
- `outputs/` — included result/reference files
