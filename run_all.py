# run_all_fixed.py - Fix virtual environment issue
import subprocess
import sys
import os
import time
import shutil
from pathlib import Path

class SentimentAnalysisSetup:
    def __init__(self):
        self.base_dir = Path(".")
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results"
        self.logs_dir = self.base_dir / "logs"
        
        # Detect Python executable (virtual environment aware)
        self.python_cmd = self.detect_python_executable()
        
    def detect_python_executable(self):
        """Tự động detect Python executable (với virtual env)"""
        
        # Check if running in virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("🔍 Detected virtual environment")
            python_cmd = sys.executable
        else:
            # Try to find virtual environment
            venv_paths = [
                "./sentiment_env/bin/python",  # Linux/Mac
                "./sentiment_env/Scripts/python.exe",  # Windows
                "./venv/bin/python",  # Alternative name
                "./venv/Scripts/python.exe"
            ]
            
            python_cmd = "python"  # Default fallback
            
            for venv_path in venv_paths:
                if Path(venv_path).exists():
                    python_cmd = venv_path
                    print(f"🔍 Found virtual environment: {venv_path}")
                    break
            else:
                print("⚠️ No virtual environment detected, using system Python")
        
        print(f"🐍 Using Python: {python_cmd}")
        return python_cmd
    
    def print_step(self, step_num, title, description=""):
        print("\n" + "="*80)
        print(f"📋 BƯỚC {step_num}: {title}")
        if description:
            print(f"💡 {description}")
        print("="*80)
    
    def run_command(self, command, description=""):
        """Chạy command với correct Python executable"""
        # Replace 'python' with correct executable
        if command.startswith('python '):
            command = command.replace('python ', f'{self.python_cmd} ', 1)
        
        print(f"🔄 {description if description else f'Executing: {command}'}")
        
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                cwd=self.base_dir
            )
            
            if result.returncode == 0:
                print(f"✅ Thành công!")
                if result.stdout.strip():
                    output_lines = result.stdout.strip().split('\n')
                    # Show last few lines of output
                    for line in output_lines[-3:]:
                        if line.strip():
                            print(f"   {line}")
                return True
            else:
                print(f"❌ Lỗi: {result.stderr}")
                # Also show stdout in case of error
                if result.stdout.strip():
                    print(f"📤 Output: {result.stdout.strip()}")
                return False
                
        except Exception as e:
            print(f"💥 Exception: {e}")
            return False
    
    def check_file_exists(self, filepath, description=""):
        """Kiểm tra file tồn tại và hiển thị size"""
        file_path = Path(filepath)
        if file_path.exists():
            size = file_path.stat().st_size
            if size > 1024*1024:  # > 1MB
                size_str = f"({size/(1024*1024):.1f} MB)"
            elif size > 1024:  # > 1KB
                size_str = f"({size/1024:.1f} KB)"
            else:
                size_str = f"({size} bytes)"
            print(f"✅ {description if description else file_path.name} - {size_str}")
            return True
        else:
            print(f"❌ {description if description else file_path.name} - Không tồn tại")
            return False
    
    def test_python_environment(self):
        """Test Python environment và packages"""
        self.print_step(1, "TEST PYTHON ENVIRONMENT", "Kiểm tra Python và packages")
        
        # Test Python executable
        test_cmd = f'{self.python_cmd} --version'
        result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Python version: {result.stdout.strip()}")
        else:
            print(f"❌ Python test failed: {result.stderr}")
            return False
        
        # Test critical packages
        test_packages = ['pandas', 'numpy', 'sklearn', 'transformers']
        
        for package in test_packages:
            test_cmd = f'{self.python_cmd} -c "import {package}; print(f\'{package}: OK\')"'
            result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {package}")
            else:
                print(f"❌ {package} - {result.stderr.strip()}")
                return False
        
        print("✅ Python environment OK!")
        return True
    
    def create_directories(self):
        """Tạo thư mục cần thiết"""
        self.print_step(2, "SETUP DIRECTORIES", "Tạo cấu trúc thư mục")
        
        directories = [self.data_dir, self.models_dir, self.results_dir, self.logs_dir]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            print(f"📁 {directory.name}/ - OK")
        
        print("✅ Directories ready!")
        return True
    
    def check_required_files(self):
        """Kiểm tra files cần thiết"""
        self.print_step(3, "CHECK FILES", "Kiểm tra script files")
        
        required_files = {
            "generate_data.py": "Data generation script",
            "train.py": "Model training script", 
            "app.py": "Flask API server"
        }
        
        all_exist = True
        for filename, description in required_files.items():
            if not self.check_file_exists(filename, description):
                all_exist = False
        
        if not all_exist:
            print("⚠️ Một số files bị thiếu. Hãy tạo chúng trước khi tiếp tục.")
            return False
        
        print("✅ All required files present!")
        return True
    
    def generate_data(self):
        """Generate training data"""
        self.print_step(4, "GENERATE DATA", "Tạo dữ liệu training")
        
        success = self.run_command(
            "python generate_data.py", 
            "Running data generation script"
        )
        
        if success:
            # Check output files
            csv_files = ["train.csv", "validation.csv", "test.csv"]
            print("\n📊 Checking output files:")
            
            all_exist = True
            total_samples = 0
            
            for csv_file in csv_files:
                filepath = self.data_dir / csv_file
                if self.check_file_exists(filepath, csv_file):
                    # Count lines in CSV
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            lines = len(f.readlines()) - 1  # Subtract header
                        print(f"     → {lines} samples")
                        total_samples += lines
                    except:
                        pass
                else:
                    all_exist = False
            
            if all_exist:
                print(f"\n📈 Total samples: {total_samples}")
                print("✅ Data generation completed!")
                return True
        
        print("❌ Data generation failed!")
        return False
    
    def train_model(self):
        """Train the model"""
        self.print_step(5, "TRAIN MODEL", "Fine-tuning sentiment model")
        
        # Backup existing model
        model_path = self.models_dir / "fine_tuned_sentiment"
        if model_path.exists():
            backup_path = self.models_dir / f"backup_model_{int(time.time())}"
            shutil.move(str(model_path), str(backup_path))
            print(f"📦 Backed up existing model to {backup_path.name}")
        
        print("⏰ Estimated time: 30-90 minutes")
        print("🔥 Starting training...")
        
        # Show progress hint
        print("💡 You can monitor progress in the terminal output")
        
        success = self.run_command(
            "python train.py",
            "Training model (this will take a while...)"
        )
        
        if success:
            print("\n🔍 Checking trained model:")
            
            # Check model files
            model_files = {
                "config.json": "Model configuration",
                "pytorch_model.bin": "Model weights", 
                "tokenizer.json": "Tokenizer",
                "tokenizer_config.json": "Tokenizer config"
            }
            
            all_exist = True
            total_size = 0
            
            for filename, description in model_files.items():
                filepath = model_path / filename
                if self.check_file_exists(filepath, description):
                    total_size += filepath.stat().st_size
                else:
                    all_exist = False
            
            if all_exist:
                print(f"\n📊 Total model size: {total_size/(1024*1024):.1f} MB")
                print("✅ Training completed successfully!")
                return True
        
        print("❌ Training failed!")
        return False
    
    def test_model(self):
        """Test the trained model"""
        self.print_step(6, "TEST MODEL", "Quick model functionality test")
        
        # Test with Python command to ensure right environment
        test_script = '''
import sys
sys.path.append(".")
from transformers import pipeline

try:
    classifier = pipeline(
        "text-classification",
        model="./models/fine_tuned_sentiment",
        tokenizer="./models/fine_tuned_sentiment"
    )
    
    test_cases = [
        ("Sản phẩm tuyệt vời!", "positive"),
        ("Chất lượng tệ!", "negative"), 
        ("Bình thường thôi.", "neutral")
    ]
    
    print("🧪 Testing model...")
    correct = 0
    
    for text, expected in test_cases:
        result = classifier(text)
        predicted = result[0]["label"]
        confidence = result[0]["score"]
        
        status = "✅" if predicted == expected else "❌"
        print(f"{status} \\"{text}\\" → {predicted} ({confidence:.3f})")
        
        if predicted == expected:
            correct += 1
    
    accuracy = correct / len(test_cases)
    print(f"\\n🎯 Quick test accuracy: {accuracy:.2%} ({correct}/{len(test_cases)})")
    
    if accuracy >= 0.67:
        print("✅ Model working well!")
    else:
        print("⚠️ Model may need improvement")
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
'''
        
        # Write test script to temp file
        with open("temp_test.py", "w", encoding="utf-8") as f:
            f.write(test_script)
        
        try:
            success = self.run_command(
                "python temp_test.py",
                "Running model test"
            )
            
            # Clean up temp file
            Path("temp_test.py").unlink(missing_ok=True)
            
            return success
            
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            Path("temp_test.py").unlink(missing_ok=True)
            return False
    
    def start_api_server(self):
        """Start the API server"""
        self.print_step(7, "START API", "Launch web interface")
        
        print("🚀 Ready to start API server")
        print("📱 URL: http://localhost:8080")
        print("💡 Press Ctrl+C to stop server")
        
        start_now = input("\n❓ Start server now? (Y/n): ").lower()
        
        if start_now in ['', 'y', 'yes']:
            print("🎯 Starting Flask server...")
            try:
                # Use correct Python executable
                subprocess.run([self.python_cmd, "app.py"], cwd=self.base_dir)
            except KeyboardInterrupt:
                print("\n🛑 Server stopped!")
        else:
            print(f"💡 Start later with: {self.python_cmd} app.py")
        
        return True
    
    def run_all_steps(self):
        """Execute all setup steps"""
        print("🚀 SENTIMENT ANALYSIS - COMPLETE SETUP")
        print("🐍 Using virtual environment aware execution")
        print("\n📋 Steps:")
        print("   1. Test Python environment")
        print("   2. Setup directories") 
        print("   3. Check required files")
        print("   4. Generate training data")
        print("   5. Train model (30-90 min)")
        print("   6. Test trained model")
        print("   7. Start API server")
        
        if input("\n❓ Continue with setup? (Y/n): ").lower() in ['n', 'no']:
            print("🚫 Setup cancelled!")
            return False
        
        steps = [
            ("Test Environment", self.test_python_environment),
            ("Create Directories", self.create_directories),
            ("Check Files", self.check_required_files),
            ("Generate Data", self.generate_data),
            ("Train Model", self.train_model),
            ("Test Model", self.test_model),
            ("Start API", self.start_api_server)
        ]
        
        try:
            for step_name, step_func in steps:
                print(f"\n⏳ Executing: {step_name}")
                success = step_func()
                
                # Critical steps that must succeed
                if not success and step_name in ["Test Environment", "Check Files", "Generate Data", "Train Model"]:
                    print(f"\n💥 Critical step '{step_name}' failed!")
                    print("🛑 Stopping execution.")
                    return False
                
                # Pause between steps (except for API server which is blocking)
                if step_name != "Start API":
                    if step_name == "Train Model":
                        print(f"✅ {step_name} completed!")
                        time.sleep(2)  # Brief pause for long operation
                    else:
                        input(f"\n⏸️ {step_name} done. Press Enter to continue...")
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Setup interrupted by user!")
            return False
        
        print("\n🎉 Setup completed successfully!")
        return True

def main():
    """Main entry point"""
    print("🤖 SENTIMENT ANALYSIS SETUP")
    print("🔧 Virtual Environment Aware Version")
    
    setup = SentimentAnalysisSetup()
    setup.run_all_steps()

if __name__ == "__main__":
    main()