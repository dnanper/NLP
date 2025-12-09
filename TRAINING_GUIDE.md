# 🚀 Hướng Dẫn Training và Evaluation

## 📊 Dataset Splits

Dữ liệu được chia thành 3 tập **KHÔNG CHỒNG CHÉO** (no data leakage):

```
data/processed/
├── train_tokenized.pkl       # Training set
├── validation_tokenized.pkl  # Validation set (dùng trong training)
└── test_tokenized.pkl         # Test set (chỉ dùng cuối cùng)
```

**✅ Đúng:**

- `train` → Training
- `validation` → Validation trong quá trình train
- `test` → Evaluation cuối cùng (KHÔNG dùng trong training)

**❌ Sai:**

- ~~Random split từ train thành train/val~~ (đã fix)
- ~~Dùng test trong training~~ (không bao giờ làm)

## 🎯 Training Workflow

### 1. Train Model (Vietnamese → English)

```bash
python trainer/train.py
```

**Quá trình:**

1. Load `train_tokenized.pkl` → Training
2. Load `validation_tokenized.pkl` → Validation (tính val_loss mỗi epoch)
3. Save best model khi val_loss thấp nhất
4. **KHÔNG** sử dụng test set

**Output:**

- `checkpoints/best_model_vi2en/best_model.pt` - Model tốt nhất
- `checkpoints/best_model_vi2en/latest.pt` - Checkpoint cuối cùng
- `checkpoints/best_model_vi2en/training_curves.png` - Biểu đồ loss

### 2. Train Bidirectional (Vietnamese ↔ English)

```bash
python trainer/train_bidirectional.py
```

**Đặc biệt:**

- Mỗi cặp (vi, en) được dùng 2 lần:
  - Lần 1: vi → en
  - Lần 2: en → vi
- Single model cho cả 2 hướng

**Output:**

- `checkpoints/best_model_bidirectional/best_model.pt`

### 3. Evaluate on Test Set (SAU KHI TRAIN XONG)

```bash
python trainer/evaluate.py
```

**Quá trình:**

1. Load trained model từ checkpoint
2. Load `test_tokenized.pkl` (lần đầu tiên dùng)
3. Translate toàn bộ test set với beam search
4. Tính BLEU score

**Output:**

- In ra BLEU-1, BLEU-2, BLEU-3, BLEU-4
- `checkpoints/best_model_vi2en/test_results.txt`
- Show 5 ví dụ translations

## 🔧 Thay Đổi Kích Cỡ Model

Trong `trainer/train.py` hoặc `trainer/train_bidirectional.py`, tìm dòng:

```python
# Model configuration - THAY ĐỔI KÍCH CỠ MODEL TẠI ĐÂY:
# .small() - 256d, 4 layers, ~60M params (fast training)
# .base()  - 512d, 6 layers, ~65M params (balanced) ⭐ RECOMMENDED
# .large() - 1024d, 6 layers, ~213M params (best quality)
# .deep()  - 512d, 12 layers (very deep)
config = TransformerConfig.base()  # <-- THAY ĐỔI Ở ĐÂY
```

**Chọn size phù hợp:**

| Size       | d_model | layers | params | Training Time | BLEU  | Recommend for             |
| ---------- | ------- | ------ | ------ | ------------- | ----- | ------------------------- |
| `.small()` | 256     | 4      | ~60M   | 6h            | 25-30 | Quick experiments, laptop |
| `.base()`  | 512     | 6      | ~65M   | 12h           | 30-35 | ⭐ Production, balanced   |
| `.large()` | 1024    | 6      | ~213M  | 24h           | 35-40 | Best quality, research    |
| `.deep()`  | 512     | 12     | ~120M  | 18h           | 32-37 | Very deep network         |

**Ví dụ thay đổi:**

```python
# For fast training (laptop, quick test)
config = TransformerConfig.small()

# For best quality (powerful GPU)
config = TransformerConfig.large()
```

## 📈 Monitoring Training

### During Training

**Training logs hiển thị:**

```
Epoch 1: 100%|████████| 1234/1234 [12:34<00:00, 1.64it/s, loss=3.4567, lr=0.000001]
Validating: 100%|████████| 123/123 [01:23<00:00, 1.48it/s, val_loss=3.2345]

============================================================
Epoch 1 Summary:
  Train Loss: 3.4567
  Val Loss:   3.2345
  LR:         0.000001
  Time:       756.2s
============================================================

✓ New best model saved! Val Loss: 3.2345
```

**Indicators:**

- ✅ **Val loss giảm** → Model đang học tốt
- ⚠️ **Val loss tăng** → Overfitting, cân nhắc early stopping
- ✅ **Train loss > Val loss** → Còn room để train
- ⚠️ **Train loss << Val loss** → Overfitting

### Training Curves

Xem biểu đồ: `checkpoints/best_model_*/training_curves.png`

**Dấu hiệu tốt:**

- Loss giảm đều qua các epochs
- Val loss theo sát train loss
- Learning rate decay smooth

**Dấu hiệu xấu:**

- Val loss tăng sớm → Reduce learning rate
- Loss không giảm → Check data/model config
- Val loss oscillate → Reduce batch size

## 🎓 Best Practices

### 1. Start Small

```bash
# Thử nghiệm với small model trước
config = TransformerConfig.small()
NUM_EPOCHS = 5
```

### 2. Monitor Validation

- Check val_loss mỗi epoch
- Save best model (tự động)
- Early stopping nếu val_loss không giảm sau 3-5 epochs

### 3. Resume Training

```python
# Trong train.py main()
trainer.train(
    NUM_EPOCHS,
    save_every=1,
    resume_from='checkpoints/best_model_vi2en/latest.pt'
)
```

### 4. Test CUỐI CÙNG

- **KHÔNG** dùng test set để tune hyperparameters
- **CHỈ** evaluate trên test set 1 lần cuối
- Use validation set để chọn model

## 📝 Example Workflow

```bash
# 1. Train model
python trainer/train.py
# → Saves to checkpoints/best_model_vi2en/best_model.pt

# 2. Monitor training
# Watch training_curves.png
# Check val_loss in logs

# 3. If need to continue training
# Edit train.py: resume_from='checkpoints/.../latest.pt'
python trainer/train.py

# 4. FINAL evaluation on test set
python trainer/evaluate.py
# → Prints BLEU scores
# → Saves test_results.txt
# → Shows sample translations

# 5. Use model for inference
from trainer import Translator
translator = Translator(
    checkpoint_path='checkpoints/best_model_vi2en/best_model.pt',
    vocab_path='SentencePiece-from-scratch/tokenizer_models/vocabulary.txt'
)
result = translator.translate("Xin chào")
```

## 🐛 Troubleshooting

### Training không giảm loss

- ✅ Check learning rate (mặc định 1e-4 là tốt)
- ✅ Verify data đúng format
- ✅ Try smaller model (.small()) trước

### Out of Memory (OOM)

- ✅ Giảm BATCH_SIZE trong `config.py`
- ✅ Dùng model nhỏ hơn (.small())
- ✅ Giảm max_len

### Overfitting

- ✅ Tăng dropout (0.1 → 0.3)
- ✅ Tăng label_smoothing (0.1 → 0.15)
- ✅ Early stopping

### BLEU score thấp

- ✅ Train longer (30+ epochs)
- ✅ Dùng model lớn hơn (.base() hoặc .large())
- ✅ Check data quality
- ✅ Use beam search trong inference (beam_size=4-6)

## 📚 Summary

**Key Points:**

1. ✅ Train với train set + validation set
2. ✅ Validation trong mỗi epoch để chọn best model
3. ✅ Test set CHỈ dùng CUỐI CÙNG để evaluate
4. ✅ Thay đổi model size bằng `.small()`, `.base()`, `.large()`
5. ✅ Monitor val_loss để detect overfitting
6. ✅ Use beam search cho inference chất lượng cao

**Files to Run:**

- `trainer/train.py` - Training
- `trainer/train_bidirectional.py` - Bidirectional training
- `test_model.py` - Final evaluation on test set

Good luck! 🚀
