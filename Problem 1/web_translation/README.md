# Web Translation - English to Vietnamese

Ứng dụng web đơn giản để dịch tiếng Anh sang tiếng Việt sử dụng model Transformer tự xây dựng.

## Cài đặt

1. Cài đặt các dependencies:

```bash
pip install -r requirements.txt
```

Hoặc cài đặt streamlit riêng:

```bash
pip install streamlit
```

## Chạy ứng dụng

Từ thư mục `Problem 1`, chạy lệnh:

```bash
streamlit run web_translation/app.py
```

Hoặc từ thư mục `web_translation`:

```bash
cd web_translation
streamlit run app.py
```

Ứng dụng sẽ tự động mở trong trình duyệt tại địa chỉ `http://localhost:8501`

## Tính năng

- 🌐 Dịch tiếng Anh sang tiếng Việt
- 📝 Giao diện đơn giản, dễ sử dụng
- 💡 Có sẵn các câu ví dụ để thử
- ⚡ Sử dụng greedy decoding để dịch nhanh
- 📊 Hiển thị thông tin model trong sidebar

## Cấu trúc file

```
web_translation/
├── app.py              # Ứng dụng Streamlit chính
├── requirements.txt    # Dependencies
└── README.md          # Hướng dẫn này
```

## Yêu cầu

- Python 3.8+
- PyTorch
- Streamlit
- Model checkpoint đã train (`checkpoints/best_model.pt`)
- Tokenizer models (`SentencePiece-from-scratch/tokenizer_models/`)
