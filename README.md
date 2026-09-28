# Dự đoán điểm trầm cảm

Dự án nghiên cứu xây dựng mô hình dự đoán tổng điểm PHQ từ dữ liệu khảo sát, kết hợp notebook phân tích/huấn luyện với ứng dụng Gradio để xem dự đoán và giải thích từng mẫu bằng SHAP.

> **Lưu ý:** Kết quả của mô hình chỉ phục vụ mục đích nghiên cứu và minh họa. Đây không phải công cụ chẩn đoán, sàng lọc hay tư vấn y tế.

## Nội dung repo

- `Copy_of_thesis.ipynb`: khám phá dữ liệu, tạo biến mục tiêu `phq_total`, phân tích CFA/SEM, đánh giá và huấn luyện mô hình.
- `gradio_shap_app.py`: giao diện Gradio để dự đoán một mẫu, xem các đóng góp SHAP và biểu diễn các đường dẫn SEM.
- `data_survey.xlsx`: dữ liệu khảo sát đầu vào của notebook.
- `artifacts/best_model_deploy.joblib`: artifact triển khai mặc định mà ứng dụng tìm kiếm.
- `artifacts/best_model.joblib`: artifact tương thích cũ.

## Yêu cầu

- Python 3.10 trở lên.
- Các thư viện Python được notebook và ứng dụng sử dụng: `gradio`, `joblib`, `matplotlib`, `numpy`, `openpyxl`, `pandas`, `scikit-learn`, `semopy`, `shap` và `catboost`.

Cài thư viện trong môi trường ảo:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install gradio joblib matplotlib numpy openpyxl pandas scikit-learn semopy shap catboost
```

Trên Windows, kích hoạt môi trường bằng `.venv\Scripts\activate`.

## Chạy notebook

Mở `Copy_of_thesis.ipynb` bằng Jupyter hoặc VS Code, chọn kernel từ môi trường đã cài thư viện, rồi chạy lần lượt các cell. Notebook đọc sheet `dataset` trong `data_survey.xlsx` và lưu artifact triển khai vào `artifacts/best_model_deploy.joblib`.

## Chạy ứng dụng

Từ thư mục gốc repo, chạy:

```bash
python gradio_shap_app.py
```

Mở địa chỉ local do Gradio in ra trong terminal. Ứng dụng tìm model theo thứ tự `artifacts/best_model_deploy.joblib`, `artifacts/best_model.joblib` và một số đường dẫn dự phòng ở thư mục gốc. Artifact mặc định đi kèm repo có thể dùng trực tiếp; nếu thay bằng artifact khác, artifact cần tương thích với cấu trúc đầu vào và metadata mà ứng dụng sử dụng.

Có thể chỉ định artifact hoặc file mẫu khác bằng biến môi trường:

```bash
MODEL_ARTIFACT=/duong/dan/toi/model.joblib SAMPLE_FILE=/duong/dan/toi/samples.csv python gradio_shap_app.py
```

`SAMPLE_FILE` có thể là CSV, Parquet hoặc Excel. Nếu không chỉ định file mẫu, ứng dụng dùng dữ liệu mẫu chứa trong artifact; nếu không có, ứng dụng tìm trong `artifacts/model_ready_samples.parquet`, `artifacts/model_ready_samples.csv`, `artifacts/samples.parquet`, `artifacts/samples.csv` rồi `data_survey.xlsx`.

## Cách đọc kết quả

- Dự đoán là giá trị PHQ do mô hình ước lượng; nếu artifact có nhãn thực tế, giao diện cũng hiển thị nhãn và sai số.
- SHAP giải thích đóng góp của các đặc trưng cho **dự đoán của một cá nhân**. Giá trị dương làm tăng dự đoán so với mức nền; giá trị âm làm giảm dự đoán.
- Các hệ số và p-value trong sơ đồ SEM mô tả mối liên hệ ở cấp độ mô hình quần thể, không phải quan hệ nhân quả và không thay thế phần giải thích SHAP ở cấp cá nhân.

## Dữ liệu và quyền riêng tư

Dữ liệu khảo sát có thể chứa thông tin nhạy cảm. Chỉ sử dụng dữ liệu khi bạn có quyền truy cập và xử lý phù hợp; tránh chia sẻ dữ liệu định danh hoặc đưa dữ liệu nhạy cảm lên dịch vụ bên ngoài.