# app.py
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from transformers import pipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load model globally
print("Đang load model...")
try:
    classifier = pipeline(
        "text-classification",
        model="./models/fine_tuned_sentiment",
        tokenizer="./models/fine_tuned_sentiment"
    )
    print("Model loaded thành công!")
except Exception as e:
    print(f"Lỗi load model: {e}")
    classifier = None

# HTML template cho web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis API</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        .form-group { margin: 20px 0; }
        label { display: block; margin-bottom: 8px; font-weight: bold; color: #555; }
        textarea { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; resize: vertical; }
        button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
        .positive { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .negative { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .neutral { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        .confidence { font-size: 12px; opacity: 0.8; }
        .examples { margin-top: 30px; }
        .example { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; cursor: pointer; border: 1px solid #e9ecef; }
        .example:hover { background: #e9ecef; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Sentiment Analysis</h1>
        <p style="text-align: center; color: #666;">Phân tích cảm xúc văn bản tiếng Việt</p>
        
        <div class="form-group">
            <label for="text">Nhập văn bản để phân tích:</label>
            <textarea id="text" rows="4" placeholder="Ví dụ: Sản phẩm này rất tuyệt vời!"></textarea>
        </div>
        
        <button onclick="analyzeText()">Phân tích cảm xúc</button>
        
        <div id="result"></div>
        
        <div class="examples">
            <h3>Ví dụ mẫu:</h3>
            <div class="example" onclick="setExample('Sản phẩm này thật tuyệt vời, tôi rất hài lòng!')">
                Sản phẩm này thật tuyệt vời, tôi rất hài lòng!
            </div>
            <div class="example" onclick="setExample('Chất lượng rất tệ, không đáng tiền')">
                Chất lượng rất tệ, không đáng tiền
            </div>
            <div class="example" onclick="setExample('Sản phẩm bình thường, không có gì đặc biệt')">
                Sản phẩm bình thường, không có gì đặc biệt
            </div>
        </div>
    </div>

    <script>
        function setExample(text) {
            document.getElementById('text').value = text;
        }
        
        function analyzeText() {
            const text = document.getElementById('text').value.trim();
            if (!text) {
                alert('Vui lòng nhập văn bản!');
                return;
            }
            
            const button = document.querySelector('button');
            button.disabled = true;
            button.textContent = 'Đang phân tích...';
            
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({text: text})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('result').innerHTML = 
                        `<div class="result" style="background: #f8d7da; color: #721c24;">Lỗi: ${data.error}</div>`;
                } else {
                    const sentiment = data.sentiment;
                    const confidence = (data.confidence * 100).toFixed(1);
                    let className = sentiment;
                    let emoji = sentiment === 'positive' ? '😊' : (sentiment === 'negative' ? '😞' : '😐');
                    
                    document.getElementById('result').innerHTML = 
                        `<div class="result ${className}">
                            <strong>${emoji} Cảm xúc: ${sentiment.toUpperCase()}</strong>
                            <div class="confidence">Độ tin cậy: ${confidence}%</div>
                        </div>`;
                }
            })
            .catch(error => {
                document.getElementById('result').innerHTML = 
                    `<div class="result" style="background: #f8d7da; color: #721c24;">Lỗi kết nối: ${error}</div>`;
            })
            .finally(() => {
                button.disabled = false;
                button.textContent = 'Phân tích cảm xúc';
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not classifier:
            return jsonify({'error': 'Model chưa được load'}), 500
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Thiếu trường "text"'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text không được để trống'}), 400
        
        # Predict
        result = classifier(text)
        
        response = {
            'text': text,
            'sentiment': result[0]['label'],
            'confidence': result[0]['score']
        }
        
        logger.info(f"Prediction: {text[:50]} -> {response['sentiment']} ({response['confidence']:.3f})")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Lỗi prediction: {str(e)}'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        if not classifier:
            return jsonify({'error': 'Model chưa được load'}), 500
        
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'Thiếu trường "texts"'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({'error': 'texts phải là một list'}), 400
        
        # Predict batch
        results = classifier(texts)
        
        response = []
        for text, result in zip(texts, results):
            response.append({
                'text': text,
                'sentiment': result['label'],
                'confidence': result['score']
            })
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': f'Lỗi batch prediction: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None
    })

# Thay đổi dòng cuối trong app.py
if __name__ == '__main__':
    print("🚀 Starting Sentiment Analysis API...")
    print("📱 Web interface: http://localhost:8080")
    print("🔗 API endpoint: http://localhost:8080/predict")
    app.run(host='0.0.0.0', port=8080, debug=False)  # Đổi từ 5000 sang 8080
