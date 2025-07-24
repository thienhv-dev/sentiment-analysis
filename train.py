# train.py
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import time

class SentimentTrainer:
    def __init__(self, model_name="distilbert-base-multilingual-cased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Label mapping
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        
    def load_data(self):
        """Load và prepare dữ liệu"""
        print("Đang load dữ liệu...")
        
        self.train_df = pd.read_csv('data/train.csv')
        self.val_df = pd.read_csv('data/validation.csv')
        self.test_df = pd.read_csv('data/test.csv')
        
        print(f"Train: {len(self.train_df)} mẫu")
        print(f"Validation: {len(self.val_df)} mẫu")
        print(f"Test: {len(self.test_df)} mẫu")
        
    def setup_model(self):
        """Setup model và tokenizer"""
        print(f"Đang load model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        print(f"Model parameters: {self.model.num_parameters():,}")
    
    def tokenize_function(self, examples):
        """Tokenize text"""
        return self.tokenizer(
            examples["text"], 
            truncation=True, 
            padding=True, 
            max_length=256  # Giảm xuống để tiết kiệm memory
        )
    
    def create_dataset(self, df):
        """Tạo dataset từ DataFrame"""
        df_copy = df.copy()
        df_copy['labels'] = df_copy['label'].map(self.label2id)
        
        dataset = Dataset.from_pandas(df_copy)
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        
        # Chỉ giữ lại các column cần thiết
        tokenized_dataset = tokenized_dataset.remove_columns(['text', 'label'])
        
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        """Tính toán metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self):
        """Fine-tune model"""
        print("Đang chuẩn bị datasets...")
        
        train_dataset = self.create_dataset(self.train_df)
        val_dataset = self.create_dataset(self.val_df)
        
        # Training arguments tối ưu cho CPU
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=2e-5,
            per_device_train_batch_size=4,  # Batch size nhỏ cho CPU
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            logging_steps=50,
            warmup_steps=100,
            no_cuda=True,  # Force sử dụng CPU
            dataloader_num_workers=0,  # Tránh multiprocessing issues
            save_total_limit=2,  # Chỉ giữ 2 checkpoint tốt nhất
            report_to=None,  # Không report lên wandb/tensorboard
        )
        
        # Tạo trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Bắt đầu training
        print("Bắt đầu fine-tuning...")
        start_time = time.time()
        
        self.trainer.train()
        
        training_time = time.time() - start_time
        print(f"Training hoàn thành trong {training_time/3600:.2f} giờ")
        
        # Lưu model
        self.trainer.save_model("./models/fine_tuned_sentiment")
        self.tokenizer.save_pretrained("./models/fine_tuned_sentiment")
        
        print("Đã lưu model thành công!")
    
    def evaluate_model(self):
        """Đánh giá model trên test set"""
        print("Đang đánh giá model...")
        
        test_dataset = self.create_dataset(self.test_df)
        
        # Predict
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Convert back to label names
        y_pred_labels = [self.id2label[pred] for pred in y_pred]
        y_true_labels = [self.id2label[true] for true in y_true]
        
        # Tính accuracy
        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        
        print(f"\nKết quả đánh giá:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        from sklearn.metrics import classification_report
        print(f"\nClassification Report:")
        print(classification_report(y_true_labels, y_pred_labels))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true_labels, y_pred_labels, 
                            labels=['negative', 'neutral', 'positive'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Neutral', 'Positive'],
                   yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy
    
    def test_predictions(self):
        """Test với một số câu mẫu"""
        print("Testing với câu mẫu...")
        
        from transformers import pipeline
        
        classifier = pipeline(
            "text-classification",
            model="./models/fine_tuned_sentiment",
            tokenizer="./models/fine_tuned_sentiment"
        )
        
        test_texts = [
            "Sản phẩm này thật tuyệt vời, tôi rất hài lòng!",
            "Chất lượng rất tệ, không đáng đồng tiền bát gạo",
            "Sản phẩm bình thường, không có gì đặc biệt",
            "Dịch vụ xuất sắc, nhân viên nhiệt tình",
            "Giao hàng chậm, đóng gói không cẩn thận",
            "Giá cả hợp lý, chất lượng ổn"
        ]
        
        print(f"\n{'='*60}")
        print("TESTING PREDICTIONS")
        print(f"{'='*60}")
        
        for text in test_texts:
            result = classifier(text)
            print(f"Text: {text}")
            print(f"Prediction: {result[0]['label']} (confidence: {result[0]['score']:.3f})")
            print("-" * 60)

def main():
    print("🚀 Bắt đầu Fine-tuning Sentiment Analysis Model")
    print("=" * 60)
    
    # Khởi tạo trainer
    trainer = SentimentTrainer(model_name="distilbert-base-multilingual-cased")
    
    # Load dữ liệu
    trainer.load_data()
    
    # Setup model
    trainer.setup_model()
    
    # Train model
    trainer.train_model()
    
    # Evaluate model
    accuracy = trainer.evaluate_model()
    
    # Test predictions
    trainer.test_predictions()
    
    print(f"\n🎉 Hoàn thành! Final Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()