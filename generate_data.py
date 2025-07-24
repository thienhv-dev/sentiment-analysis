# generate_data.py
import pandas as pd
import random
from sklearn.model_selection import train_test_split

def create_vietnamese_reviews():
    """Tạo dữ liệu đánh giá khách hàng tiếng Việt"""
    
    positive_reviews = [
        "Sản phẩm này thật tuyệt vời, tôi rất hài lòng!",
        "Sản phẩm tuyệt vời, chất lượng rất tốt!",
        "Dịch vụ khách hàng xuất sắc, nhân viên nhiệt tình",
        "Giao hàng nhanh chóng, đóng gói cẩn thận",
        "Giá cả hợp lý so với chất lượng sản phẩm",
        "Rất hài lòng với trải nghiệm mua sắm",
        "Sản phẩm đúng như mô tả, chất lượng cao",
        "Nhân viên tư vấn chuyên nghiệp và thân thiện",
        "Thời gian giao hàng nhanh hơn dự kiến",
        "Chất lượng vượt mong đợi, sẽ mua lại",
        "Dịch vụ sau bán hàng tốt, hỗ trợ nhanh chóng",
        "Sản phẩm đóng gói đẹp mắt, sang trọng",
        "Giá thành phải chăng, phù hợp túi tiền",
        "Chất liệu cao cấp, bền đẹp theo thời gian",
        "Thiết kế hiện đại, phù hợp với xu hướng",
        "Tính năng đa dạng, đáp ứng nhu cầu",
        "Hướng dẫn sử dụng rõ ràng, dễ hiểu",
        "Bảo hành dài hạn, yên tâm sử dụng",
        "Thương hiệu uy tín, đáng tin cậy",
        "Màu sắc đẹp, phù hợp với sở thích",
        "Kích thước chuẩn, vừa vặn hoàn hảo"
    ]
    
    negative_reviews = [
        "Sản phẩm kém chất lượng, không như quảng cáo",
        "Dịch vụ khách hàng tệ, không giải quyết vấn đề",
        "Giao hàng chậm trễ, không đúng hẹn",
        "Giá đắt so với chất lượng thực tế",
        "Sản phẩm bị lỗi ngay từ đầu",
        "Nhân viên thiếu chuyên nghiệp, thái độ không tốt",
        "Đóng gói không cẩn thận, sản phẩm bị hỏng",
        "Chất lượng giảm so với lần mua trước",
        "Không đúng mô tả, khác hẳn hình ảnh",
        "Dịch vụ sau bán hàng kém, không hỗ trợ",
        "Màu sắc phai, không bền theo thời gian",
        "Kích thước không đúng, quá nhỏ hoặc quá to",
        "Tính năng không hoạt động như mô tả",
        "Bảo hành kém, khó khăn khi cần hỗ trợ",
        "Thiết kế lỗi thời, không hấp dẫn",
        "Chất liệu rẻ tiền, dễ hỏng",
        "Hướng dẫn không rõ ràng, khó hiểu",
        "Giá thành quá cao, không xứng đáng",
        "Thời gian chờ đợi quá lâu",
        "Trải nghiệm tồi tệ, không muốn quay lại",
        "Sản phẩm tuyệt vời, không có gì đặc biệt",
    ]
    
    neutral_reviews = [
        "Sản phẩm bình thường, không có gì đặc biệt",
        "Chất lượng ổn, đáp ứng nhu cầu cơ bản",
        "Giá cả hợp lý, chấp nhận được",
        "Dịch vụ tạm được, không xuất sắc lắm",
        "Giao hàng đúng hẹn, đóng gói bình thường",
        "Nhân viên thái độ bình thường",
        "Sản phẩm tạm ổn, sử dụng được",
        "Chất lượng trung bình, không tệ lắm",
        "Thiết kế bình thường, không nổi bật",
        "Tính năng cơ bản, đủ dùng",
        "Màu sắc ổn, không quá ấn tượng",
        "Kích thước tiêu chuẩn, vừa phải",
        "Bảo hành bình thường, theo tiêu chuẩn",
        "Hướng dẫn đầy đủ nhưng không quá chi tiết",
        "Trải nghiệm tạm được, không có complaint",
        "Chất liệu ổn, sử dụng bình thường",
        "Dịch vụ đủ tốt, đáp ứng mong đợi",
        "Sản phẩm đúng mô tả, không có gì bất ngờ",
        "Thời gian giao hàng chấp nhận được",
        "Nhìn chung ổn, có thể cân nhắc mua"
    ]
    
    # Tạo dataset cân bằng
    data = []
    
    # Mỗi loại 334 mẫu để có tổng cộng ~1000 mẫu
    for i in range(17):  # Lặp để tạo đủ 334 mẫu mỗi loại
        for review in positive_reviews:
            if len(data) < 334:
                data.append({
                    "text": f"{review} Tôi {random.choice(['rất', 'thực sự', 'cực kỳ'])} {random.choice(['hài lòng', 'ấn tượng', 'thích'])}.",
                    "label": "positive"
                })
        
        for review in negative_reviews:
            if len([d for d in data if d['label'] == 'negative']) < 334:
                data.append({
                    "text": f"{review} Tôi {random.choice(['rất', 'thực sự', 'cực kỳ'])} {random.choice(['thất vọng', 'không hài lòng', 'bức xúc'])}.",
                    "label": "negative"
                })
        
        for review in neutral_reviews:
            if len([d for d in data if d['label'] == 'neutral']) < 332:
                data.append({
                    "text": f"{review} Nhìn chung {random.choice(['ổn', 'tạm được', 'chấp nhận được'])}.",
                    "label": "neutral"
                })
        
        if len(data) >= 1000:
            break
    
    return data[:1000]

def main():
    print("Đang tạo dữ liệu...")
    
    # Tạo dữ liệu
    data = create_vietnamese_reviews()
    df = pd.DataFrame(data)
    
    print(f"Tổng số mẫu: {len(df)}")
    print(df['label'].value_counts())
    
    # Chia dữ liệu train/validation/test
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )
    
    # Lưu dữ liệu
    train_df.to_csv('data/train.csv', index=False, encoding='utf-8')
    val_df.to_csv('data/validation.csv', index=False, encoding='utf-8')
    test_df.to_csv('data/test.csv', index=False, encoding='utf-8')
    
    print(f"\nĐã lưu dữ liệu:")
    print(f"- Train: {len(train_df)} mẫu")
    print(f"- Validation: {len(val_df)} mẫu") 
    print(f"- Test: {len(test_df)} mẫu")

if __name__ == "__main__":
    main()
