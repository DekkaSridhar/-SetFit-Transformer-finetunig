# Sentiment Classification using SetFit

This project demonstrates a complete pipeline to train, evaluate, and deploy a **SetFit** (Sentence Transformer Fine-Tuning) model for **multi-class sentiment classification** on a custom dataset. It uses the `SetFit` library for few-shot learning and is suitable for classifying sentences into one of four sentiment categories: `NEGATIVE`, `POSITIVE`, `NEUTRAL`, `MIXED`.

## 📦 Dependencies

Install required packages using:

```bash
pip install setfit torch --index-url https://download.pytorch.org/whl/cu118
pip install datasets pandas seaborn matplotlib scikit-learn
```

## 📁 Dataset

The dataset is assumed to be a CSV file named `Dataset.csv` with at least two columns:
- `sentence`: The input text
- `label`: One of the four string labels (`NEGATIVE`, `POSITIVE`, `NEUTRAL`, `MIXED`)

## 🚀 Pipeline Overview

### 1. Load and Prepare Dataset
- Load the dataset using `pandas`
- Convert it into a HuggingFace `Dataset`
- Perform an 80-20 train-validation split
- Map string labels to integers for model compatibility

### 2. Load Pretrained SetFit Model
- Uses `BAAI/bge-base-en-v1.5` transformer as backbone
- Customizes it for 4 sentiment labels

### 3. Training
- The model is trained with configurable parameters using `SetFit.Trainer`
- Uses oversampling to handle class imbalance
- Logs progress, evaluates during training, and saves checkpoints

### 4. Evaluation
- Evaluates on validation set using accuracy, confusion matrix, and classification report (precision, recall, F1-score)
- Visualizes the confusion matrix using `seaborn`

### 5. Inference & Export
- Run predictions on custom sample inputs
- Predict sentiment for the full dataset
- Adds predicted labels and exports results to `predicted_results.csv`

## 🧠 Example Predictions

```python
sample_sentences = [
    "More locations would be helpful...",
    "I deeply appreciate the care...",
    "Disappointed in the consult...",
    "A little more heads up on schedule...",
    "Excellent compassionate staff",
    "Appointments were very unorganized..."
]
preds = model.predict(sample_sentences)
print(preds)
```

## 📉 Evaluation Example

```python
# Confusion Matrix and Classification Report
from sklearn.metrics import confusion_matrix, classification_report
```

Confusion matrix is visualized for true vs. predicted sentiment classes.

## 🗂 Outputs

- `saved_setfit_model/` - Directory containing the fine-tuned model
- `saved_setfit_model.zip` - Downloadable zip of the trained model
- `predicted_results.csv` - Dataset with an added "Predicted Label" column

## 🛠 Training Config Snapshot

```python
args = TrainingArguments(
    output_dir='checkpoints',
    batch_size=(16, 2),
    max_steps=800,
    num_epochs=3,
    sampling_strategy="oversampling",
    num_iterations=5,
    end_to_end=True,
    body_learning_rate=(2e-5, 1e-5),
    head_learning_rate=4e-2,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=100,
    metric_for_best_model="eval_embedding_loss",
    load_best_model_at_end=True
)
```

## 📎 Notes

- The training pipeline is designed to work efficiently in **Google Colab**
- Label mapping is essential: 
  ```python
  label_mapping = {
      "NEGATIVE": 0,
      "POSITIVE": 1,
      "NEUTRAL": 2,
      "MIXED": 3
  }
  ```
- Supports additional inference after training on both sample and full datasets.