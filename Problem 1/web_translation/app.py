"""
Streamlit Web Application for English to Vietnamese Translation
Using custom Transformer model
"""

import streamlit as st
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models_best import BestTransformer
from utils.data_processing import DataProcessor
from config import Config


@st.cache_resource
def load_model():
    """Load and cache the translation model"""
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
    TOKENIZER_DIR = PROJECT_ROOT / "SentencePiece-from-scratch" / "tokenizer_models"
    DEVICE = 'cpu'  # Force CPU for inference
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint['config']
    config.device = DEVICE
    
    # Load tokenizer
    processor = DataProcessor(Config)
    processor.load_tokenizer(str(TOKENIZER_DIR))
    
    # Create model
    model = BestTransformer(
        src_vocab_size=processor.vocab_size,
        tgt_vocab_size=processor.vocab_size,
        config=config
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    return model, processor, config, DEVICE


def translate_sentence(text: str, model, processor, device) -> str:
    """Translate a single sentence from English to Vietnamese"""
    if not text.strip():
        return ""
    
    # Tokenize
    src_ids = processor.encode_sentence(text, add_sos=True, add_eos=True)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    
    # Translate (greedy decoding - fastest)
    with torch.no_grad():
        tgt_ids = model.translate_greedy(src_tensor)
    
    # Decode
    translation = processor.decode_sentence(tgt_ids, skip_special_tokens=True)
    
    return translation


def main():
    # Page configuration
    st.set_page_config(
        page_title="English to Vietnamese Translator",
        page_icon="🌐",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-title {
            text-align: center;
            color: #1E88E5;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-bottom: 30px;
        }
        .stTextArea textarea {
            font-size: 16px;
        }
        .translation-box {
            background-color: #f0f7ff;
            border-radius: 10px;
            padding: 20px;
            margin-top: 10px;
            border-left: 4px solid #1E88E5;
        }
        .translation-label {
            color: #1E88E5;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .translation-text {
            font-size: 18px;
            color: #333;
            line-height: 1.6;
        }
        .info-box {
            background-color: #e8f5e9;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown("<h1 class='main-title'>🌐 English to Vietnamese Translator</h1>", unsafe_allow_html=True)
    # st.markdown("<p class='subtitle'>Powered by Custom Transformer Model</p>", unsafe_allow_html=True)
    
    # Load model with spinner
    with st.spinner("🔄 Loading translation model... Please wait..."):
        try:
            model, processor, config, device = load_model()
            model_loaded = True
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            model_loaded = False
    
    if model_loaded:
        # Success message (only shown once)
        if 'model_loaded_msg_shown' not in st.session_state:
            st.success("✅ Model loaded successfully!")
            st.session_state.model_loaded_msg_shown = True
        
        # Input section
        st.markdown("### 📝 Enter English text:")
        
        # Text input
        input_text = st.text_area(
            label="English text",
            placeholder="Type or paste your English text here...",
            height=150,
            label_visibility="collapsed",
            key="input_text"
        )
        
        # Translate button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            translate_btn = st.button(
                "🔄 Translate",
                type="primary",
                use_container_width=True
            )
        
        # Translation result
        if translate_btn and input_text.strip():
            with st.spinner("🔄 Translating..."):
                translation = translate_sentence(input_text, model, processor, device)
            
            st.markdown("### Vietnamese Translation:")
            st.markdown(f"""
                <div class='translation-box'>
                    <div class='translation-text'>{translation}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Copy button
            # st.code(translation, language=None)
        
        elif translate_btn and not input_text.strip():
            st.warning("⚠️ Please enter some text to translate.")
        
        # Model info in sidebar
        with st.sidebar:
            st.markdown("### ℹ️ Model Information")
            st.markdown(f"""
            - **Model**: Custom Transformer
            - **Device**: {device}
            - **Parameters**: {model.count_parameters():,}
            - **Vocabulary Size**: {processor.vocab_size:,}
            """)
            
            st.markdown("---")
            st.markdown("### 📖 About")
            st.markdown("""
            This translator uses a custom-built Transformer model 
            trained on the PhoMT dataset for English-Vietnamese translation.
            
            **Features:**
            - Greedy decoding for fast translation
            - Custom SentencePiece tokenizer
            - Attention-based neural machine translation
            """)
    
    # # Footer
    # st.markdown("---")
    # st.markdown(
    #     "<p style='text-align: center; color: #888; font-size: 12px;'>"
    #     "Built with ❤️ using Streamlit and PyTorch"
    #     "</p>",
    #     unsafe_allow_html=True
    # )


if __name__ == "__main__":
    main()
