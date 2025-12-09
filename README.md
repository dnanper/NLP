# Vietnamese-English Translator

Complete Neural Machine Translation system using Transformer architecture.

## 📁 Project Structure

```
ViEn_Translator/
│
├── models_best/              # 🏗️ MODEL ARCHITECTURE ONLY
│   ├── __init__.py
│   ├── README.md
│   ├── config.py            # TransformerConfig
│   ├── transformer.py       # BestTransformer (main model)
│   ├── encoder.py           # Pre-LN encoder
│   ├── decoder.py           # Pre-LN decoder
│   ├── attention.py         # Multi-head attention
│   ├── feed_forward.py      # Feed-forward network
│   ├── embeddings.py        # Embedding layers
│   ├── positional_encoding.py  # Position encoding
│   ├── layer_norm.py        # LayerNorm & RMSNorm
│   ├── beam_search.py       # Beam search decoder
│   └── label_smoothing.py   # Label smoothing loss
│
├── trainer/                  # 🚀 TRAINING & INFERENCE
│   ├── __init__.py
│   ├── train.py             # Training script (vi→en)
│   ├── train_bidirectional.py  # Bidirectional (vi↔en)
│   ├── inference.py         # Translation inference
│   └── evaluate.py          # BLEU evaluation
│
├── utils/                    # 🛠️ DATA PROCESSING
│   ├── __init__.py
│   └── data_processing.py   # DataProcessor, Dataset, collate_fn
│
├── SentencePiece-from-scratch/  # 📝 TOKENIZER
│   ├── tokenizer_models/
│   │   ├── vocabulary.txt   # 32k vocab
│   │   └── metadata.txt
│   └── ...
│
├── data/                     # 📊 DATASETS
│   └── processed/
│       ├── train_tokenized.pkl
│       ├── validation_tokenized.pkl
│       └── test_tokenized.pkl
│
├── checkpoints/              # 💾 SAVED MODELS
│   ├── best_model_vi2en/
│   └── best_model_bidirectional/
│
├── config.py                 # Global config
└── README.md                 # This file
```

## ✨ Features

### Model Architecture

- ✅ **Pre-Layer Normalization** - More stable training
- ✅ **Weight Tying** - Decoder embedding = output projection
- ✅ **Label Smoothing** - Better generalization (0.1)
- ✅ **Beam Search** - High-quality inference with length penalty
- ✅ **Multi-Query Attention** - Faster inference (optional)
- ✅ **Gradient Clipping** - Prevent gradient explosion

### Training Features

- ✅ **Warmup LR Scheduler** - Linear warmup + inverse sqrt decay
- ✅ **Mixed Precision Training** - Faster with modern GPUs
- ✅ **Checkpoint Management** - Auto-save best model
- ✅ **Training Curves** - Automatic plotting
- ✅ **Resume Training** - Load from checkpoint

### Data Processing

- ✅ **SentencePiece Tokenizer** - 32,000 BPE tokens
- ✅ **Cached Tokenization** - Fast data loading (.pkl files)
- ✅ **Proper Masking** - Padding mask + causal mask
- ✅ **Bidirectional Support** - Train single model for both directions

## 🚀 Quick Start

### 1. Train Vietnamese → English

```bash
python trainer/train.py
```

**Configuration:**

- Model: Base (512d, 6 layers, 65M params)
- Batch size: From `config.Config.BATCH_SIZE`
- Max length: From `config.Config.MAX_LEN`
- Device: Auto-detect CUDA/CPU
- Saves to: `checkpoints/best_model_vi2en/`

### 2. Train Bidirectional (Vietnamese ↔ English)

```bash
python trainer/train_bidirectional.py
```

**Key advantage:** Single model handles BOTH directions by using each (vi, en) pair twice:

- First time: vi → en
- Second time: en → vi

**Saves to:** `checkpoints/best_model_bidirectional/`

### 3. Inference (Translation)

```python
from trainer import Translator

translator = Translator(
    checkpoint_path='checkpoints/best_model_vi2en/best_model.pt',
    vocab_path='SentencePiece-from-scratch/tokenizer_models/vocabulary.txt'
)

# Translate a sentence
result = translator.translate("Xin chào thế giới")
print(result)  # "Hello world"

# Translate with beam search
result = translator.translate("Tôi yêu học máy", beam_size=5)
print(result)
```

### 4. Evaluate BLEU Score

```python
from trainer import Evaluator

evaluator = Evaluator(
    checkpoint_path='checkpoints/best_model_vi2en/best_model.pt',
    vocab_path='SentencePiece-from-scratch/tokenizer_models/vocabulary.txt'
)

# Evaluate on test set
bleu_score = evaluator.evaluate_file(
    src_file='data/processed/test.vi',
    tgt_file='data/processed/test.en'
)
print(f"BLEU: {bleu_score:.2f}")
```

## 📊 Model Configurations

### Small (Fast Training)

```python
config = TransformerConfig.small()
# 256d, 4 layers, ~60M params
# Good for: Quick experiments, limited GPU
```

### Base (Recommended)

```python
config = TransformerConfig.base()
# 512d, 6 layers, ~65M params
# Good for: Production, balanced quality/speed
```

### Large (Best Quality)

```python
config = TransformerConfig.large()
# 1024d, 6 layers, ~213M params
# Good for: Maximum quality, research
```

### Deep (Very Deep Network)

```python
config = TransformerConfig.deep()
# 512d, 12 layers
# Good for: Complex language pairs
```

## 🔧 Advanced Usage

### Resume Training

```python
# In train.py main() function
trainer.train(
    NUM_EPOCHS,
    save_every=1,
    resume_from='checkpoints/best_model_vi2en/latest.pt'
)
```

### Custom Configuration

```python
from models_best import TransformerConfig

config = TransformerConfig(
    d_model=512,
    n_encoder_layers=6,
    n_decoder_layers=6,
    n_heads=8,
    d_ff=2048,
    dropout=0.1,
    max_len=512,
    learning_rate=1e-4,
    warmup_steps=8000,
    label_smoothing=0.1
)
```

### Export for Production

```python
# Export model to TorchScript
model = BestTransformer(...)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scripted_model = torch.jit.script(model)
scripted_model.save('model_production.pt')
```

## 📈 Training Tips

1. **Start with small model** to verify pipeline works
2. **Monitor validation loss** - stop if overfitting
3. **Use beam search** for inference (beam_size=4-6)
4. **Bidirectional training** gives better results with same data
5. **Gradient clipping** at 1.0 prevents instability
6. **Label smoothing** 0.1 is optimal for most cases

## 🎯 Expected Results

| Model | BLEU (vi→en) | BLEU (en→vi) | Training Time |
| ----- | ------------ | ------------ | ------------- |
| Small | ~25-30       | ~23-28       | ~6 hours      |
| Base  | ~30-35       | ~28-33       | ~12 hours     |
| Large | ~35-40       | ~33-38       | ~24 hours     |

_On single GPU (RTX 3090), ~1.3M training pairs_

## 📝 Data Format

### Input Files

- `data/processed/train_tokenized.pkl` - Training data
- `data/processed/validation_tokenized.pkl` - Validation data
- `data/processed/test_tokenized.pkl` - Test data

### Format (Pickle)

```python
{
    'en': List[List[int]],  # English token IDs
    'vi': List[List[int]]   # Vietnamese token IDs
}
```

### Special Tokens

- PAD: 0
- UNK: 1
- SOS: 2 (Start of Sequence)
- EOS: 3 (End of Sequence)

## 🛠️ Dependencies

```bash
pip install torch torchvision torchaudio
pip install sentencepiece
pip install tqdm matplotlib
```

## 📚 References

- **Attention Is All You Need** - Vaswani et al. (2017)
- **Pre-LN Transformer** - Xiong et al. (2020)
- **Label Smoothing** - Szegedy et al. (2016)
- **SentencePiece** - Kudo & Richardson (2018)

## 🤝 Contributing

Feel free to:

- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📄 License

MIT License - See LICENSE file for details

---

**Happy Translating! 🌍✨**
