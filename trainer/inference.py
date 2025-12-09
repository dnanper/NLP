"""
Inference Script for Vietnamese-English Translation

Translate sentences or files using trained model
"""

import torch
from pathlib import Path
import pickle
from typing import List, Optional
import time

from models_best import BestTransformer, TransformerConfig


class Translator:
    """
    Translator for inference
    """
    
    def __init__(self, checkpoint_path: str, vocab_path: str, device: str = 'cuda'):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary.txt
            device: Device to run on
        """
        print("Loading translator...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.config = checkpoint['config']
        self.config.device = device
        
        # Load vocabulary
        self.vocab = self._load_vocab(vocab_path)
        self.id2token = {i: token for token, i in self.vocab.items()}
        
        # Create model
        vocab_size = len(self.vocab)
        self.model = BestTransformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            config=self.config
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded from {checkpoint_path}")
        print(f"  Device: {device}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    def _load_vocab(self, vocab_path: str) -> dict:
        """Load vocabulary from file"""
        vocab = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) == 2:
                    token_id, token = parts
                    vocab[token] = int(token_id)
        
        # Add special tokens if not present
        if '<PAD>' not in vocab:
            vocab['<PAD>'] = 0
        if '<UNK>' not in vocab:
            vocab['<UNK>'] = 1
        if '<SOS>' not in vocab:
            vocab['<SOS>'] = 2
        if '<EOS>' not in vocab:
            vocab['<EOS>'] = 3
        
        return vocab
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text to token IDs
        
        Note: This is a simple word-level tokenizer
        In production, use your trained BPE tokenizer
        """
        # Simple whitespace tokenization
        tokens = text.strip().split()
        
        # Add BOS and EOS
        token_ids = [self.vocab.get('<SOS>', 2)]
        
        for token in tokens:
            token_id = self.vocab.get(token, self.vocab.get('<UNK>', 1))
            token_ids.append(token_id)
        
        token_ids.append(self.vocab.get('<EOS>', 3))
        
        return token_ids
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            token = self.id2token.get(token_id, '<UNK>')
            if token not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def translate(self, text: str, beam_size: int = 5, length_penalty: float = 0.6,
                  use_greedy: bool = False) -> str:
        """
        Translate a single sentence
        
        Args:
            text: Input text
            beam_size: Beam size for beam search
            length_penalty: Length penalty factor
            use_greedy: Use greedy search instead of beam search
        
        Returns:
            Translated text
        """
        # Tokenize input
        src_ids = self.tokenize(text)
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.config.device)
        
        # Translate
        start_time = time.time()
        
        with torch.no_grad():
            if use_greedy:
                tgt_ids = self.model.translate_greedy(src_tensor)
            else:
                tgt_ids = self.model.translate_beam(
                    src_tensor, 
                    beam_size=beam_size,
                    length_penalty=length_penalty
                )
        
        translation_time = time.time() - start_time
        
        # Detokenize
        translation = self.detokenize(tgt_ids)
        
        print(f"Translation time: {translation_time:.3f}s")
        
        return translation
    
    def translate_file(self, input_file: str, output_file: str,
                       beam_size: int = 5, use_greedy: bool = False):
        """
        Translate a file line by line
        
        Args:
            input_file: Input file path
            output_file: Output file path
            beam_size: Beam size
            use_greedy: Use greedy search
        """
        print(f"Translating {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
        
        translations = []
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                translations.append('')
                continue
            
            translation = self.translate(line, beam_size, use_greedy=use_greedy)
            translations.append(translation)
            
            if i % 100 == 0:
                print(f"  Translated {i}/{len(lines)} lines")
        
        # Save translations
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for translation in translations:
                f_out.write(translation + '\n')
        
        print(f"✓ Translations saved to {output_file}")


def main():
    """Interactive translation"""
    
    # Get project root directory (parent of trainer/)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Configuration (relative to project root)
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model" / "best_model.pt"
    VOCAB_PATH = PROJECT_ROOT / "SentencePiece-from-scratch" / "tokenizer_models" / "vocabulary.txt"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create translator
    translator = Translator(CHECKPOINT_PATH, VOCAB_PATH, DEVICE)
    
    print("\n" + "=" * 60)
    print("English-Vietnamese Translator (EN → VI)")
    print("=" * 60)
    print("Commands:")
    print("  - Type a sentence to translate")
    print("  - Type 'file <input> <output>' to translate a file")
    print("  - Type 'beam <size>' to change beam size (default: 5)")
    print("  - Type 'greedy' to toggle greedy search")
    print("  - Type 'exit' to quit")
    print("=" * 60)
    
    beam_size = 5
    use_greedy = False
    
    while True:
        try:
            user_input = input("\nInput: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            elif user_input.lower().startswith('file '):
                # Translate file
                parts = user_input.split()
                if len(parts) != 3:
                    print("Usage: file <input_file> <output_file>")
                    continue
                
                input_file = parts[1]
                output_file = parts[2]
                translator.translate_file(input_file, output_file, beam_size, use_greedy)
            
            elif user_input.lower().startswith('beam '):
                # Change beam size
                try:
                    beam_size = int(user_input.split()[1])
                    print(f"Beam size set to {beam_size}")
                except:
                    print("Invalid beam size")
            
            elif user_input.lower() == 'greedy':
                # Toggle greedy search
                use_greedy = not use_greedy
                print(f"Greedy search: {'ON' if use_greedy else 'OFF'}")
            
            else:
                # Translate sentence
                print(f"\nTranslating (beam_size={beam_size}, greedy={use_greedy})...")
                translation = translator.translate(user_input, beam_size, use_greedy=use_greedy)
                print(f"\nTranslation: {translation}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
