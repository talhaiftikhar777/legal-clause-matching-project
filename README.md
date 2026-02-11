Clause Similarity Detection using Siamese BiLSTM and Attention Encoder
📌 Project Overview

This project implements and compares Siamese BiLSTM and Attention-based encoder models to determine the similarity between legal clauses. It processes a dataset of clauses, generates positive and negative pairs, trains deep learning models, evaluates performance metrics, and saves the results for reporting.

🗂 Dataset

The dataset is provided as a ZIP file containing CSV files.

Each CSV file corresponds to a clause category.

The code reads all CSVs, extracts the clause column, and assigns the category automatically.

⚙ Workflow

Read ZIP Dataset
Extract CSV files, normalize column names, and merge all clauses into a single dataframe.

Generate Clause Pairs

Positive pairs: clauses from the same category.

Negative pairs: clauses from different categories.

Ratio of negative to positive pairs is configurable.

Text Tokenization

Tokenize clauses using Tokenizer from Keras.

Pad sequences to a maximum length (MAX_SEQ_LEN = 100).

Model Definitions

Siamese BiLSTM: baseline model using bidirectional LSTM and L1 distance.

Attention Encoder: adds a custom attention layer on top of BiLSTM.

Training

Train both models with binary_crossentropy loss.

Small epochs (3-5) used for quick training; batch size = 64.

Evaluation

Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.

Compare both models using a performance table.

Visualization

Training vs validation loss and accuracy plotted using Matplotlib.

Results

Final results saved as model_comparison_results.csv.

📦 Requirements
pandas==2.1.1
numpy==1.25.2
scikit-learn==1.3.2
tensorflow==2.13.0
matplotlib==3.8.0

Install dependencies with:

pip install -r requirements.txt
🧠 Model Performance (Example)
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Siamese BiLSTM	0.9685	0.9735	0.9633	0.9684	0.9886
Attention Encoder	0.8037	0.8607	0.7248	0.7869	0.9123

Observation: Siamese BiLSTM outperforms the Attention Encoder on this dataset.
