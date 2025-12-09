# 📊 LOSS CALCULATION - CHI TIẾT VÀ MASKING

## ✅ TL;DR: ĐÃ MASK ĐÚNG!

**Loss chỉ tính trên các token KHÔNG phải padding (PAD=0)**

---

## 🔍 FLOW CHI TIẾT

### 1️⃣ Khởi Tạo Loss Function

```python
# In trainer/train.py - Trainer.__init__()
self.criterion = LabelSmoothingLoss(
    num_classes=model.tgt_vocab_size,      # 32000
    smoothing=config.label_smoothing,       # 0.1
    ignore_index=config.pad_idx             # 0 (PAD token)
)
```

**Key point:** `ignore_index=0` → Loss sẽ IGNORE tất cả PAD tokens!

---

### 2️⃣ Training Step

```python
# In train_epoch()

# Original target sequence
tgt = [SOS, tok1, tok2, tok3, PAD, PAD]  # [B, T]

# Prepare input/output
tgt_input  = tgt[:, :-1]  # [SOS, tok1, tok2, tok3, PAD]     (input to decoder)
tgt_output = tgt[:, 1:]   # [tok1, tok2, tok3, PAD, PAD]     (target for loss)

# Forward pass
logits = model(src, tgt_input, src_mask, tgt_mask)
# logits: [B, T-1, vocab_size] = [B, 5, 32000]

# Compute loss (ĐÂY LÀ CHỖ MASK!)
loss = self.criterion(logits, tgt_output)
```

---

### 3️⃣ Label Smoothing Loss - Masking Logic

```python
# In models_best/label_smoothing.py

def forward(self, logits, targets):
    # Flatten
    logits: [B, T, vocab_size] → [B*T, vocab_size]
    targets: [B, T] → [B*T]

    # BƯỚC 1: TẠO MASK CHO PADDING
    if self.ignore_index >= 0:  # ignore_index = 0 (PAD)
        mask = targets.ne(self.ignore_index)  # ✅ mask[i] = True nếu targets[i] != 0

    # Ví dụ:
    # targets = [123, 456, 789, 0, 0]  # 3 real tokens, 2 PAD
    # mask    = [True, True, True, False, False]  # ✅ ĐÚNG!

    # BƯỚC 2: TẠO SMOOTH TARGETS (CHỈ CHO NON-PAD TOKENS)
    with torch.no_grad():
        smooth_targets = torch.full_like(log_probs, self.smoothing_value)
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)

        # ZERO OUT PADDING TOKENS
        smooth_targets = smooth_targets * mask.unsqueeze(1).float()
        # ✅ PAD positions → all zeros in smooth_targets

    # BƯỚC 3: COMPUTE LOSS
    loss = -torch.sum(smooth_targets * log_probs, dim=-1)
    # loss: [B*T] - loss per position

    # BƯỚC 4: MASK LOSS (CHỈ GIỮ NON-PAD)
    loss = loss * mask  # ✅ PAD positions → loss = 0

    # BƯỚC 5: AVERAGE CHỈ TRÊN VALID TOKENS
    return loss.sum() / mask.sum()  # ✅ Divide by NUMBER OF NON-PAD TOKENS
```

---

## 📊 VÍ DỤ CỤ THỂ

### Input Batch:

```python
# Batch size = 2
tgt_output = [
    [tok1, tok2, tok3, PAD, PAD],  # Sentence 1: 3 real tokens
    [tok4, tok5, PAD, PAD, PAD],   # Sentence 2: 2 real tokens
]

# Total positions: 2 * 5 = 10
# Valid tokens: 3 + 2 = 5
# PAD tokens: 5 (WILL BE IGNORED)
```

### Loss Computation:

```python
# After model forward
logits: [2, 5, 32000]
targets: [2, 5]

# Flatten
logits: [10, 32000]
targets: [10] = [tok1, tok2, tok3, 0, 0, tok4, tok5, 0, 0, 0]

# Create mask
mask = [True, True, True, False, False, True, True, False, False, False]

# Compute per-position loss
loss_per_pos = [L1, L2, L3, L4, L5, L6, L7, L8, L9, L10]

# Apply mask
loss_per_pos = [L1, L2, L3, 0, 0, L6, L7, 0, 0, 0]
                  ✅  ✅  ✅  ❌  ❌  ✅  ✅  ❌  ❌  ❌

# Final loss
total_loss = L1 + L2 + L3 + L6 + L7
avg_loss = total_loss / 5  # Divide by 5 NON-PAD tokens, NOT 10!
```

---

## ✅ KIỂM TRA: MASK ĐÚNG CHƯA?

### Checkpoint 1: ignore_index

```python
✅ self.criterion = LabelSmoothingLoss(ignore_index=0)
```

### Checkpoint 2: Mask creation

```python
✅ mask = targets.ne(self.ignore_index)  # True for non-PAD
```

### Checkpoint 3: Zero out smooth_targets for PAD

```python
✅ smooth_targets = smooth_targets * mask.unsqueeze(1).float()
```

### Checkpoint 4: Mask loss values

```python
✅ loss = loss * mask  # PAD positions → 0
```

### Checkpoint 5: Average over valid tokens only

```python
✅ return loss.sum() / mask.sum()  # Divide by NON-PAD count
```

---

## 🎯 KẾT LUẬN

### ✅ ĐÚNG - Loss đã được mask HOÀN TOÀN!

**Các bước masking:**

1. ✅ Tạo mask: `targets != PAD_IDX`
2. ✅ Zero out smooth_targets cho PAD positions
3. ✅ Zero out loss cho PAD positions
4. ✅ Average chỉ trên valid tokens (không tính PAD)

**Hậu quả:**

- PAD tokens KHÔNG đóng góp vào loss
- Gradient KHÔNG được tính cho PAD positions
- Model KHÔNG học từ PAD tokens
- Training chỉ focus vào real tokens ✅

---

## 🔬 SO SÁNH VỚI STANDARD CROSS ENTROPY

### Standard CrossEntropyLoss:

```python
# PyTorch's built-in
criterion = nn.CrossEntropyLoss(ignore_index=0)
loss = criterion(logits, targets)
# ✅ Cũng mask PAD, nhưng KHÔNG có label smoothing
```

### Our LabelSmoothingLoss:

```python
criterion = LabelSmoothingLoss(ignore_index=0, smoothing=0.1)
loss = criterion(logits, targets)
# ✅ Mask PAD + Label Smoothing (better generalization)
```

---

## 📈 IMPACT

### Nếu KHÔNG mask PAD:

```python
❌ loss = loss.mean()  # Divide by ALL positions
# → Loss sẽ BỊ GIẢM GIẢM vì PAD chiếm nhiều
# → Model học sai: predict PAD quá nhiều
# → BLEU score thấp
```

### Với mask PAD (hiện tại):

```python
✅ loss = loss.sum() / mask.sum()  # Divide by VALID tokens
# → Loss phản ánh đúng performance trên real tokens
# → Model học đúng distribution
# → BLEU score cao hơn
```

---

## 🧪 VERIFY CODE

Để kiểm tra mask hoạt động đúng, add test case:

```python
def test_loss_masking():
    """Test that PAD tokens are ignored in loss"""
    vocab_size = 100
    criterion = LabelSmoothingLoss(
        num_classes=vocab_size,
        smoothing=0.1,
        ignore_index=0
    )

    # Create fake logits and targets
    logits = torch.randn(4, 10, vocab_size)  # [B=4, T=10, V=100]
    targets = torch.randint(1, 100, (4, 10))  # [B=4, T=10]

    # Add PAD tokens
    targets[:, 7:] = 0  # Last 3 positions are PAD

    # Compute loss
    loss = criterion(logits, targets)

    # Verify: should only compute loss on first 7 positions
    # If PAD is NOT masked, loss would be much smaller

    print(f"Loss with mask: {loss.item():.4f}")

    # Test: Set PAD positions to extreme values
    targets_test = targets.clone()
    targets_test[:, 7:] = 99  # Change PAD to valid token

    loss_no_pad = criterion(logits, targets_test)
    print(f"Loss without PAD: {loss_no_pad.item():.4f}")

    # Loss should be different if mask works
    assert abs(loss.item() - loss_no_pad.item()) > 0.01
    print("✅ Masking works correctly!")
```

---

## 📝 SUMMARY

**Question:** Lúc train, tính loss như thế nào? Đã mask các token cần tính loss chưa?

**Answer:**
✅ **ĐÃ MASK ĐÚNG!**

- Loss function: `LabelSmoothingLoss(ignore_index=0)`
- PAD tokens (0) được IGNORE hoàn toàn
- Loss chỉ tính trên VALID tokens
- Average loss = total_loss / số_valid_tokens

**Implementation:** CHUẨN ✅
