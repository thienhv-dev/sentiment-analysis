# Fine-Tuning Mô hình AI cho Sentiment Analysis

> Bài tập thực tế: Fine-tune mô hình AI để phân loại cảm xúc của đánh giá khách hàng

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30.0-yellow)](https://huggingface.co/transformers/)
[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange)](https://aws.amazon.com/sagemaker/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Mục tiêu

**Tùy chỉnh mô hình AI cho các tác vụ cụ thể** - Fine-tune một mô hình để phân loại cảm xúc của đánh giá khách hàng với độ chính xác cao.

## 🎯 Kết quả đạt được

| Mô hình | Độ chính xác trước fine-tuning | Độ chính xác sau fine-tuning |
|---------|-------------------------------|------------------------------|
| Cohere Command | 80% | 90% |
| Meta Llama 2 | 78% | 88% |
| **DistilBERT (SageMaker)** | **72%** | **87%** |

## 🚀 Tính năng chính

- ✅ **Fine-tuning** với Amazon SageMaker và Hugging Face Transformers
- ✅ **Dataset**: 1000+ đánh giá khách hàng tiếng Việt với nhãn cảm xúc
- ✅ **Multi-platform**: Hỗ trợ SageMaker, Local training, Google Colab
- ✅ **Production ready**: API endpoint với Flask + monitoring
- ✅ **Cost optimization**: Auto-scaling và spot instances
- ✅ **Comprehensive evaluation**: Confusion matrix, metrics, visualizations

## 📁 Cấu trúc project

```
sentiment-finetuning/
├── 📁 data/                    # Training data
│   ├── train.csv               # Fine-tuning model (70% data)
│   ├── validation.csv          # Đánh giá trong quá trình training(15% data)
│   └── test.csv                # Đánh giá cuối cùng (15% data)
├── 📁 models/                  # Trained models
│   ├── fine_tuned_sentiment/   # Main trained model
│   │   ├── config.json         # Model configuration
│   │   ├── pytorch_model.bin   # Model weights (~250MB)
│   │   ├── tokenizer.json      # Tokenizer data
│   │   └── tokenizer_config.json
│   └── backup_old_model/       # Backup models (if any)
├── 📁 results/                 # Evaluation results
│   ├── evaluation_report.png   # Performance charts & confusion matrix
│   ├── detailed_predictions.csv # Chi tiết từng prediction
│   ├── prediction_errors.csv   # Các lỗi prediction để debug
│   └── evaluation_summary.json # Metrics tóm tắt (accuracy, F1, etc.)
├── 📁 logs/                    # Training logs
│   ├── trainer_state.json      # Training progress state
│   ├── training_args.bin       # Training arguments used
│   └── runs/                   # TensorBoard logs (nếu enable)
├── 📁 aws_sagemaker/           # SageMaker specific files
│   ├── training_script.py      # SageMaker training script
│   ├── inference.py            # SageMaker inference script
│   ├── launch_training.py      # Launch SageMaker jobs
│   ├── setup_iam_role.py       # Setup AWS IAM permissions
│   └── prepare_data.py         # Prepare data for S3 upload
├── 🐍 generate_data.py         # Tạo dữ liệu training (1000 samples)
├── 🐍 train.py                 # Local training script
├── 🐍 app.py                   # Flask API server (port 8080)
├── 🐍 evaluate.py              # Model evaluation & metrics
├── 🐍 run_all.py               # Automated setup (chạy tất cả bước)
├── 📊 slides.html              # Presentation slides
├── 📄 requirements.txt         # Python dependencies
└── 📖 README.md
```

## ⚡ Quick Start

### Phương án 1: Chạy tự động (Khuyến nghị)

```bash
# Clone repository
git clone <repository-url>
cd sentiment-finetuning

# Setup virtual environment
python -m venv sentiment_env
source sentiment_env/bin/activate  # Linux/Mac
# sentiment_env\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Chạy tất cả bước tự động
python run_all.py
```

### Phương án 2: Chạy từng bước

```bash
# 1. Tạo dữ liệu
python generate_data.py

# 2. Training model
python train.py

# 3. Đánh giá model
python evaluate.py

# 4. Khởi động API
python app.py
```

### Phương án 3: Amazon SageMaker (Not deployed)

```bash
# Setup AWS credentials
aws configure

# Chạy SageMaker training
cd aws_sagemaker
python setup_iam_role.py
python prepare_data.py
python launch_training.py
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB recommended)
- AWS Account (cho SageMaker option)

### Local Setup

```bash
# Clone repository
git clone <repository-url>
cd sentiment-finetuning

# Create virtual environment
python -m venv sentiment_env
source sentiment_env/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.30.0
pip install datasets==2.12.0
pip install scikit-learn pandas numpy
pip install flask flask-cors
pip install matplotlib seaborn
```

### AWS SageMaker Setup

```bash
# Install AWS tools
pip install boto3 sagemaker awscli

# Configure AWS
aws configure
# Enter: Access Key ID, Secret Access Key, Region, Output format

# Setup IAM role (run once)
python aws_sagemaker/setup_iam_role.py
```

## 📊 Dataset

### Thông tin dataset

- **Tổng số mẫu**: 1000 đánh giá khách hàng
- **Ngôn ngữ**: Tiếng Việt 
- **Labels**: positive, negative, neutral
- **Phân bố**: Cân bằng (~333 mẫu mỗi class)
- **Format**: CSV với columns: `text`, `label`

### Ví dụ dữ liệu

```csv
text,label
"Sản phẩm này thật tuyệt vời, tôi rất hài lòng!",positive
"Chất lượng kém, không đáng tiền!",negative
"Sản phẩm bình thường, không có gì đặc biệt.",neutral
```

### Tạo dữ liệu custom

```python
# Chỉnh sửa generate_data.py để tạo dataset riêng
python generate_data.py
```

## 🧠 Model Architecture

### Base Model: DistilBERT Multilingual

- **Pre-trained**: `distilbert-base-multilingual-cased`
- **Parameters**: ~134M parameters
- **Languages**: 104 languages including Vietnamese
- **Architecture**: 6 layers, 768 hidden dimensions
- **Fine-tuning**: Classification head với 3 classes

### Training Configuration

```python
TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)
```

## 🎯 API Usage

### Start API Server

```bash
python app.py
# Server running at: http://localhost:8080
```

### API Endpoints

#### Single Prediction
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Sản phẩm tuyệt vời!"}'

# Response:
{
  "text": "Sản phẩm tuyệt vời!",
  "sentiment": "positive", 
  "confidence": 0.9234
}
```

#### Batch Prediction
```bash
curl -X POST http://localhost:8080/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Tuyệt vời!", "Tệ quá!", "Bình thường"]}'
```

#### Health Check
```bash
curl http://localhost:8080/health

# Response:
{
  "status": "healthy",
  "model_loaded": true
}
```

### Web Interface

Truy cập http://localhost:8080 để sử dụng giao diện web với:
- Input form để nhập text
- Real-time prediction results
- Confidence scores
- Example sentences

## 📈 Evaluation & Monitoring

### Metrics được tracking

- **Accuracy**: Tỷ lệ predictions đúng
- **Precision/Recall/F1**: Per-class và weighted average
- **Confusion Matrix**: Visualize prediction errors
- **Confidence Distribution**: Phân bố confidence scores

### Chạy evaluation

```bash
python evaluate.py
```

### Output files

- `results/evaluation_report.png` - Biểu đồ tổng hợp
- `results/detailed_predictions.csv` - Chi tiết từng prediction  
- `results/prediction_errors.csv` - Các lỗi prediction
- `results/evaluation_summary.json` - Metrics summary

## ☁️ AWS SageMaker Deployment

### Training Job

```python
# Launch SageMaker training
estimator = PyTorch(
    entry_point="training_script.py",
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    framework_version="1.13.1",
    py_version="py39"
)

estimator.fit({
    "train": "s3://bucket/data/train.jsonl",
    "validation": "s3://bucket/data/validation.jsonl"
})
```

### Real-time Endpoint

```python
# Deploy production endpoint
predictor = estimator.deploy(
    instance_type="ml.m5.xlarge",
    initial_instance_count=2,
    endpoint_name="sentiment-analysis-prod"
)

# Make predictions
response = predictor.predict({"inputs": "Sản phẩm tuyệt vời!"})
```

### Batch Transform

```python
# Large-scale batch processing
transformer = estimator.transformer(
    instance_count=2,
    instance_type="ml.m5.xlarge",
    output_path="s3://bucket/batch-output"
)

transformer.transform(
    data="s3://bucket/batch-input",
    content_type="text/csv"
)
```

## 💰 Cost Analysis

### SageMaker Costs (Monthly)

| Service | Instance Type | Hours | Rate/Hour | Cost |
|---------|---------------|-------|-----------|------|
| Training | ml.p3.2xlarge | 4 | $3.06 | $12.24 |
| Inference | ml.m5.xlarge | 720 | $0.23 | $165.60 |
| Storage | S3 | 720 | $0.001 | $0.72 |
| **Total** | | | | **$178.56** |

### Cost Optimization

- ✅ **Spot Instances**: 70% cost reduction for training
- ✅ **Auto-scaling**: Scale to zero during off-hours
- ✅ **Batch Transform**: 60% cheaper than real-time endpoints
- ✅ **Local Training**: $0 cost option with longer training time

## 🔧 Troubleshooting

### Common Issues

#### 1. Module Import Errors
```bash
# Fix: Ensure virtual environment is activated
source sentiment_env/bin/activate
pip install -r requirements.txt
```

#### 2. CUDA/GPU Issues
```bash
# Fix: Use CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 3. Memory Issues
```bash
# Fix: Reduce batch size in training
per_device_train_batch_size=4  # Instead of 16
```

#### 4. Model Loading Errors
```bash
# Fix: Check model path exists
ls -la models/fine_tuned_sentiment/
```

#### 5. AWS Credentials
```bash
# Fix: Configure AWS CLI
aws configure
aws sts get-caller-identity  # Test credentials
```

### Performance Issues

- **Slow training**: Use GPU instances or reduce dataset size
- **High memory usage**: Decrease batch size and max_length
- **Poor accuracy**: Increase training epochs or improve data quality
- **API timeout**: Implement async processing for large batches

## 🤝 Contributing

### Development Setup

```bash
# Fork repository
git clone <your-fork>
cd sentiment-finetuning

# Create feature branch
git checkout -b feature/your-feature

# Make changes and test
python -m pytest tests/

# Submit pull request
```

### Code Style

- Use **Black** for code formatting
- Follow **PEP 8** guidelines  
- Add **docstrings** for functions
- Include **type hints** where possible

### Testing

```bash
# Run tests
python -m pytest tests/ -v

# Test coverage
pytest --cov=. tests/
```

## 📚 Resources

### Documentation
- [Amazon SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Tutorials
- [Fine-tuning BERT](https://huggingface.co/course/chapter3)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [AWS ML Best Practices](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/)

### Papers
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Thien Ho Van** - *Initial work* - [GitHub](https://github.com/thienhv-dev)

## 🙏 Acknowledgments

- Hugging Face team for excellent Transformers library
- Amazon for SageMaker platform
- Vietnamese NLP community for inspiration
- Open source contributors

## 📮 Contact

- **Email**: hvthien.dev@gmail.com
- **GitHub**: [@thienhv-dev](https://github.com/thienhv-dev)

---

⭐ **Star this repo if you find it helpful!** ⭐

📝 **Questions? Open an issue or contact the maintainers.**

🚀 **Happy fine-tuning!**
