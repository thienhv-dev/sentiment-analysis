# evaluate.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_curve, auc
)
import numpy as np
import warnings
import os
from collections import Counter
import time

warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, model_path="./models/fine_tuned_sentiment"):
        self.model_path = model_path
        self.classifier = None
        self.test_df = None
        self.predictions = None
        self.confidences = None
        
    def load_model_and_data(self):
        """Load model và test data"""
        print("🔄 Loading model và test data...")
        
        # Load model
        try:
            self.classifier = pipeline(
                "text-classification",
                model=self.model_path,
                tokenizer=self.model_path
            )
            print("✅ Model loaded thành công!")
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
            return False
        
        # Load test data
        try:
            self.test_df = pd.read_csv('data/test.csv')
            print(f"✅ Test data loaded: {len(self.test_df)} mẫu")
            print("Label distribution:")
            print(self.test_df['label'].value_counts())
        except Exception as e:
            print(f"❌ Lỗi load test data: {e}")
            return False
        
        return True
    
    def predict_test_set(self):
        """Predict trên toàn bộ test set"""
        print("\n🔄 Đang predict test set...")
        
        predictions = []
        confidences = []
        
        start_time = time.time()
        
        for idx, text in enumerate(self.test_df['text']):
            if idx % 20 == 0:
                print(f"Processed {idx}/{len(self.test_df)} samples...")
            
            try:
                result = self.classifier(text)
                predictions.append(result[0]['label'])
                confidences.append(result[0]['score'])
            except Exception as e:
                print(f"Lỗi predict sample {idx}: {e}")
                predictions.append('neutral')  # Default prediction
                confidences.append(0.33)
        
        end_time = time.time()
        
        self.predictions = predictions
        self.confidences = confidences
        
        # Add predictions to dataframe
        self.test_df['predicted'] = predictions
        self.test_df['confidence'] = confidences
        
        print(f"✅ Prediction hoàn thành trong {end_time - start_time:.2f} giây")
        print(f"⚡ Tốc độ: {len(self.test_df)/(end_time - start_time):.1f} samples/second")
    
    def calculate_metrics(self):
        """Tính toán các metrics chi tiết"""
        print("\n📊 Calculating metrics...")
        
        true_labels = self.test_df['label']
        pred_labels = self.test_df['predicted']
        
        # Overall accuracy
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Per-class accuracy
        print("\n📈 Per-class Accuracy:")
        for label in ['positive', 'negative', 'neutral']:
            mask = true_labels == label
            if mask.sum() > 0:
                class_acc = accuracy_score(true_labels[mask], pred_labels[mask])
                print(f"   {label.capitalize():8}: {class_acc:.4f} ({class_acc*100:.2f}%)")
        
        # Precision, Recall, F1 cho từng class
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, average=None, 
            labels=['positive', 'negative', 'neutral']
        )
        
        print("\n📋 Detailed Metrics per Class:")
        print(f"{'Class':10} {'Precision':10} {'Recall':10} {'F1-Score':10} {'Support':10}")
        print("-" * 55)
        
        for i, label in enumerate(['positive', 'negative', 'neutral']):
            print(f"{label:10} {precision[i]:10.3f} {recall[i]:10.3f} {f1[i]:10.3f} {support[i]:10}")
        
        # Weighted averages
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted'
        )
        
        print("-" * 55)
        print(f"{'Weighted':10} {precision_avg:10.3f} {recall_avg:10.3f} {f1_avg:10.3f} {len(true_labels):10}")
        
        # Classification report
        print(f"\n📄 Detailed Classification Report:")
        print(classification_report(true_labels, pred_labels))
        
        return accuracy, precision_avg, recall_avg, f1_avg
    
    def create_visualizations(self):
        """Tạo các biểu đồ phân tích"""
        print("\n🎨 Creating visualizations...")
        
        # Tạo thư mục results nếu chưa có
        os.makedirs('results', exist_ok=True)
        
        # Setup matplotlib
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (15, 12)
        plt.rcParams['font.size'] = 10
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Evaluation Report', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        ax1 = axes[0, 0]
        cm = confusion_matrix(self.test_df['label'], self.test_df['predicted'], 
                            labels=['positive', 'negative', 'neutral'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Positive', 'Negative', 'Neutral'],
                   yticklabels=['Positive', 'Negative', 'Neutral'])
        ax1.set_title('Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. Confidence Distribution
        ax2 = axes[0, 1]
        ax2.hist(self.test_df['confidence'], bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        ax2.set_title('Confidence Score Distribution')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.axvline(np.mean(self.test_df['confidence']), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.test_df["confidence"]):.3f}')
        ax2.legend()
        
        # 3. Confidence by Predicted Class
        ax3 = axes[0, 2]
        colors = ['green', 'red', 'orange']
        for i, label in enumerate(['positive', 'negative', 'neutral']):
            mask = self.test_df['predicted'] == label
            if mask.sum() > 0:
                ax3.hist(self.test_df[mask]['confidence'], bins=15, alpha=0.6, 
                        label=label.capitalize(), color=colors[i], edgecolor='black')
        ax3.set_title('Confidence by Predicted Class')
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # 4. Accuracy by Confidence Threshold
        ax4 = axes[1, 0]
        thresholds = np.arange(0.3, 1.0, 0.05)
        accuracies = []
        sample_counts = []
        
        for threshold in thresholds:
            high_conf_mask = self.test_df['confidence'] >= threshold
            if high_conf_mask.sum() > 0:
                acc = accuracy_score(
                    self.test_df[high_conf_mask]['label'],
                    self.test_df[high_conf_mask]['predicted']
                )
                accuracies.append(acc)
                sample_counts.append(high_conf_mask.sum())
            else:
                accuracies.append(0)
                sample_counts.append(0)
        
        ax4.plot(thresholds, accuracies, marker='o', color='blue', label='Accuracy')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(thresholds, sample_counts, marker='s', color='red', alpha=0.7, label='Sample Count')
        
        ax4.set_title('Accuracy vs Confidence Threshold')
        ax4.set_xlabel('Confidence Threshold')
        ax4.set_ylabel('Accuracy', color='blue')
        ax4_twin.set_ylabel('Sample Count', color='red')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        # 5. Error Analysis - Confidence of Errors
        ax5 = axes[1, 1]
        errors = self.test_df[self.test_df['label'] != self.test_df['predicted']]
        correct = self.test_df[self.test_df['label'] == self.test_df['predicted']]
        
        ax5.hist(errors['confidence'], bins=15, alpha=0.7, label=f'Errors ({len(errors)})', 
                color='red', edgecolor='black')
        ax5.hist(correct['confidence'], bins=15, alpha=0.5, label=f'Correct ({len(correct)})', 
                color='green', edgecolor='black')
        ax5.set_title('Confidence Distribution: Errors vs Correct')
        ax5.set_xlabel('Confidence Score')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        
        # 6. Performance Summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Tính toán summary metrics
        accuracy = accuracy_score(self.test_df['label'], self.test_df['predicted'])
        precision_avg = precision_recall_fscore_support(
            self.test_df['label'], self.test_df['predicted'], average='weighted'
        )[0]
        recall_avg = precision_recall_fscore_support(
            self.test_df['label'], self.test_df['predicted'], average='weighted'
        )[1]
        f1_avg = precision_recall_fscore_support(
            self.test_df['label'], self.test_df['predicted'], average='weighted'
        )[2]
        
        # Summary text
        summary_text = f"""
Performance Summary

Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)
Weighted Precision: {precision_avg:.3f}
Weighted Recall: {recall_avg:.3f}
Weighted F1-Score: {f1_avg:.3f}

Sample Distribution:
• Total Samples: {len(self.test_df)}
• Correct Predictions: {len(correct)} ({len(correct)/len(self.test_df)*100:.1f}%)
• Wrong Predictions: {len(errors)} ({len(errors)/len(self.test_df)*100:.1f}%)

Confidence Statistics:
• Mean Confidence: {np.mean(self.test_df['confidence']):.3f}
• Std Confidence: {np.std(self.test_df['confidence']):.3f}
• Min Confidence: {np.min(self.test_df['confidence']):.3f}
• Max Confidence: {np.max(self.test_df['confidence']):.3f}

Low Confidence Predictions (<0.7): {len(self.test_df[self.test_df['confidence'] < 0.7])}
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('results/evaluation_report.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved to 'results/evaluation_report.png'")
    
    def analyze_errors(self):
        """Phân tích chi tiết các lỗi prediction"""
        print("\n🔍 Error Analysis...")
        
        errors = self.test_df[self.test_df['label'] != self.test_df['predicted']].copy()
        
        if len(errors) == 0:
            print("🎉 Không có lỗi prediction nào!")
            return
        
        print(f"📊 Tổng số lỗi: {len(errors)}/{len(self.test_df)} ({len(errors)/len(self.test_df)*100:.2f}%)")
        
        # Error by class
        print("\n📈 Lỗi theo từng class:")
        for true_label in ['positive', 'negative', 'neutral']:
            class_errors = errors[errors['label'] == true_label]
            total_class = len(self.test_df[self.test_df['label'] == true_label])
            
            if total_class > 0:
                error_rate = len(class_errors) / total_class
                print(f"   {true_label.capitalize():8}: {len(class_errors):3}/{total_class:3} errors ({error_rate*100:.1f}%)")
                
                if len(class_errors) > 0:
                    print(f"      Predicted as: {Counter(class_errors['predicted']).most_common()}")
        
        # Low confidence errors
        low_conf_errors = errors[errors['confidence'] < 0.7]
        print(f"\n🔻 Low confidence errors (<0.7): {len(low_conf_errors)}")
        
        # High confidence errors (tự tin nhưng sai)
        high_conf_errors = errors[errors['confidence'] >= 0.8]
        print(f"⚠️  High confidence errors (>=0.8): {len(high_conf_errors)}")
        
        if len(high_conf_errors) > 0:
            print("\n🚨 Một số high confidence errors:")
            print("-" * 80)
            for idx, row in high_conf_errors.head(5).iterrows():
                print(f"Text: {row['text'][:60]}...")
                print(f"True: {row['label']}, Predicted: {row['predicted']}, Confidence: {row['confidence']:.3f}")
                print("-" * 80)
        
        # Most common error patterns
        print(f"\n📋 Error patterns:")
        error_patterns = errors.groupby(['label', 'predicted']).size().sort_values(ascending=False)
        for (true_label, pred_label), count in error_patterns.head(5).items():
            print(f"   {true_label} → {pred_label}: {count} cases")
        
        # Save detailed errors
        errors_detailed = errors[['text', 'label', 'predicted', 'confidence']].copy()
        errors_detailed = errors_detailed.sort_values('confidence', ascending=False)
        errors_detailed.to_csv('results/prediction_errors.csv', index=False, encoding='utf-8')
        print(f"\n💾 Chi tiết errors đã lưu vào 'results/prediction_errors.csv'")
    
    def performance_comparison(self):
        """So sánh performance với baseline"""
        print("\n🏆 Performance Comparison...")
        
        accuracy = accuracy_score(self.test_df['label'], self.test_df['predicted'])
        
        # Random baseline
        random_accuracy = 1/3  # 3 classes
        
        # Most frequent class baseline
        most_frequent = self.test_df['label'].mode()[0]
        majority_accuracy = len(self.test_df[self.test_df['label'] == most_frequent]) / len(self.test_df)
        
        print(f"📊 Performance Comparison:")
        print(f"   Random Baseline:      {random_accuracy:.3f} ({random_accuracy*100:.1f}%)")
        print(f"   Majority Baseline:    {majority_accuracy:.3f} ({majority_accuracy*100:.1f}%)")
        print(f"   Fine-tuned Model:     {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   Improvement over Random: +{(accuracy - random_accuracy)*100:.1f}%")
        print(f"   Improvement over Majority: +{(accuracy - majority_accuracy)*100:.1f}%")
        
        # Expected accuracy ranges
        print(f"\n🎯 Model Performance Assessment:")
        if accuracy >= 0.90:
            print("   🌟 Excellent! (≥90%)")
        elif accuracy >= 0.85:
            print("   ⭐ Very Good! (85-89%)")
        elif accuracy >= 0.80:
            print("   ✅ Good (80-84%)")
        elif accuracy >= 0.75:
            print("   👍 Acceptable (75-79%)")
        elif accuracy >= 0.70:
            print("   ⚠️  Needs Improvement (70-74%)")
        else:
            print("   ❌ Poor Performance (<70%)")
    
    def save_detailed_results(self):
        """Lưu kết quả chi tiết"""
        print("\n💾 Saving detailed results...")
        
        # Save predictions with confidence
        results_df = self.test_df[['text', 'label', 'predicted', 'confidence']].copy()
        results_df = results_df.sort_values('confidence', ascending=False)
        results_df.to_csv('results/detailed_predictions.csv', index=False, encoding='utf-8')
        
        # Save summary stats
        summary_stats = {
            'total_samples': len(self.test_df),
            'accuracy': accuracy_score(self.test_df['label'], self.test_df['predicted']),
            'mean_confidence': np.mean(self.test_df['confidence']),
            'std_confidence': np.std(self.test_df['confidence']),
            'correct_predictions': len(self.test_df[self.test_df['label'] == self.test_df['predicted']]),
            'wrong_predictions': len(self.test_df[self.test_df['label'] != self.test_df['predicted']]),
            'low_confidence_count': len(self.test_df[self.test_df['confidence'] < 0.7])
        }
        
        import json
        with open('results/evaluation_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
        print("✅ Results saved:")
        print("   📄 results/detailed_predictions.csv")
        print("   📊 results/evaluation_summary.json") 
        print("   🎨 results/evaluation_report.png")
    
    def run_complete_evaluation(self):
        """Chạy evaluation hoàn chỉnh"""
        print("🚀 Starting Complete Model Evaluation")
        print("=" * 60)
        
        # Load model and data
        if not self.load_model_and_data():
            print("❌ Không thể load model hoặc data. Evaluation dừng.")
            return False
        
        # Predict test set
        self.predict_test_set()
        
        # Calculate metrics
        accuracy, precision, recall, f1 = self.calculate_metrics()
        
        # Create visualizations
        self.create_visualizations()
        
        # Analyze errors
        self.analyze_errors()
        
        # Performance comparison
        self.performance_comparison()
        
        # Save results
        self.save_detailed_results()
        
        print("\n" + "=" * 60)
        print("🎉 Evaluation Complete!")
        print(f"🎯 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("📁 Check 'results/' folder for detailed reports")
        
        return True

def main():
    """Main function để chạy evaluation"""
    evaluator = ModelEvaluator()
    success = evaluator.run_complete_evaluation()
    
    if success:
        print("\n✨ Model evaluation hoàn thành thành công!")
    else:
        print("\n❌ Model evaluation thất bại!")

if __name__ == "__main__":
    main()