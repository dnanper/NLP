"""
Evaluation Script for Translation Quality

Compute BLEU score and other metrics
"""

import torch
from torch.utils.data import DataLoader
from typing import List, Dict
import time
from tqdm import tqdm
from collections import Counter
import math
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models_best import BestTransformer, TransformerConfig
from utils.data_processing import DataProcessor, collate_fn
from config import Config


def compute_bleu(references: List[List[int]], hypotheses: List[List[int]], 
                 max_n: int = 4) -> Dict[str, float]:
    """
    Compute BLEU score
    
    Args:
        references: List of reference token sequences
        hypotheses: List of hypothesis token sequences
        max_n: Maximum n-gram order
    
    Returns:
        Dictionary with BLEU scores
    """
    assert len(references) == len(hypotheses)
    
    total_count = [0] * max_n
    clip_count = [0] * max_n
    ref_length = 0
    hyp_length = 0
    
    for ref, hyp in zip(references, hypotheses):
        ref_length += len(ref)
        hyp_length += len(hyp)
        
        # Count n-grams
        for n in range(1, max_n + 1):
            # Reference n-grams
            ref_ngrams = Counter([tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)])
            
            # Hypothesis n-grams
            hyp_ngrams = Counter([tuple(hyp[i:i+n]) for i in range(len(hyp) - n + 1)])
            
            # Clip count
            for ngram, count in hyp_ngrams.items():
                clip_count[n-1] += min(count, ref_ngrams.get(ngram, 0))
            
            # Total count
            total_count[n-1] += max(len(hyp) - n + 1, 0)
    
    # Compute precision for each n-gram
    precisions = []
    for i in range(max_n):
        if total_count[i] > 0:
            p = clip_count[i] / total_count[i]
            precisions.append(p)
        else:
            precisions.append(0.0)
    
    # Brevity penalty
    if hyp_length > ref_length:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_length / hyp_length) if hyp_length > 0 else 0.0
    
    # Compute BLEU for each n-gram order
    results = {}
    for n in range(1, max_n + 1):
        if min(precisions[:n]) > 0:
            log_precisions = sum(math.log(p) for p in precisions[:n]) / n
            bleu_n = bp * math.exp(log_precisions)
        else:
            bleu_n = 0.0
        results[f'bleu-{n}'] = bleu_n * 100
    
    # Add other metrics
    results.update({
        'bp': bp,
        'precision_1': precisions[0] * 100,
        'precision_2': precisions[1] * 100,
        'precision_3': precisions[2] * 100,
        'precision_4': precisions[3] * 100,
        'ref_length': ref_length,
        'hyp_length': hyp_length,
    })
    
    return results


class Evaluator:
    """
    Evaluate translation model
    """
    
    def __init__(self, model: BestTransformer, test_loader: DataLoader):
        """
        Args:
            model: Trained model
            test_loader: Test dataloader
        """
        self.model = model
        self.test_loader = test_loader
        self.device = model.device
    
    @torch.no_grad()
    def evaluate(self, use_beam: bool = True, beam_size: int = 5) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            use_beam: Use beam search (slower but better)
            beam_size: Beam size
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        references = []
        hypotheses = []
        
        print(f"Evaluating on test set...")
        print(f"  Method: {'Beam Search' if use_beam else 'Greedy Search'}")
        if use_beam:
            print(f"  Beam size: {beam_size}")
        
        start_time = time.time()
        
        for batch in tqdm(self.test_loader, desc="Translating"):
            src = batch['src'].to(self.device)  # [batch_size, src_len]
            tgt = batch['tgt'].to(self.device)  # [batch_size, tgt_len]
            
            # Get reference (remove BOS and EOS)
            for t in tgt:
                ref_tokens = t.tolist()
                # Remove padding
                ref_tokens = [tok for tok in ref_tokens if tok != 0]
                # Remove BOS (2) and EOS (3)
                ref_tokens = [tok for tok in ref_tokens if tok not in [2, 3]]
                references.append(ref_tokens)
            
            # Translate
            if use_beam:
                # Translate each in batch
                for i in range(src.size(0)):
                    translation = self.model.translate_beam(
                        src[i:i+1],
                        max_len=self.model.config.max_len,
                        beam_size=beam_size,
                        length_penalty=0.6
                    )
                    # Remove special tokens
                    hyp_tokens = [tok for tok in translation.tolist() if tok not in [0, 2, 3]]
                    hypotheses.append(hyp_tokens)
            else:
                # Greedy search
                for i in range(src.size(0)):
                    translation = self.model.translate_greedy(
                        src[i:i+1],
                        max_len=self.model.config.max_len
                    )
                    hyp_tokens = [tok for tok in translation.tolist() if tok not in [0, 2, 3]]
                    hypotheses.append(hyp_tokens)
        
        eval_time = time.time() - start_time
        
        # Compute BLEU
        bleu_results = compute_bleu(references, hypotheses)
        
        # Add timing info
        bleu_results['eval_time'] = eval_time
        bleu_results['sentences_per_sec'] = len(references) / eval_time
        
        # Print results
        print("\n" + "=" * 60)
        print("Evaluation Results:")
        print("=" * 60)
        print(f"BLEU-1:            {bleu_results['bleu-1']:.2f}")
        print(f"BLEU-2:            {bleu_results['bleu-2']:.2f}")
        print(f"BLEU-3:            {bleu_results['bleu-3']:.2f}")
        print(f"BLEU-4:            {bleu_results['bleu-4']:.2f}")
        print(f"Brevity Penalty:   {bleu_results['bp']:.4f}")
        print(f"Ref length:        {bleu_results['ref_length']:,}")
        print(f"Hyp length:        {bleu_results['hyp_length']:,}")
        print(f"Eval time:         {eval_time:.1f}s")
        print(f"Speed:             {bleu_results['sentences_per_sec']:.1f} sent/s")
        print("=" * 60)
        
        return bleu_results
    
    def show_examples(self, processor: DataProcessor, num_examples: int = 5):
        """Show translation examples"""
        self.model.eval()
        
        print("\n" + "=" * 60)
        print(f"Translation Examples:")
        print("=" * 60)
        
        count = 0
        for batch in self.test_loader:
            if count >= num_examples:
                break
            
            src = batch['src'][0:1].to(self.device)  # Take first sample
            tgt = batch['tgt'][0:1].to(self.device)
            
            # Translate
            translation = self.model.translate_beam(
                src,
                max_len=self.model.config.max_len,
                beam_size=4,
                length_penalty=0.6
            )
            
            # Decode to text
            src_text = processor.decode_sentence(src[0].tolist(), skip_special_tokens=True)
            ref_text = processor.decode_sentence(tgt[0].tolist(), skip_special_tokens=True)
            pred_text = processor.decode_sentence(translation.tolist(), skip_special_tokens=True)
            
            print(f"\nExample {count+1}:")
            print(f"  Source:      {src_text}")
            print(f"  Reference:   {ref_text}")
            print(f"  Translation: {pred_text}")
            
            count += 1


def main():
    """Main evaluation function - Evaluate on TEST SET"""
    
    # ========== Configuration ==========
    
    # Get project root directory (parent of trainer/)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # CHANGE THIS to your trained model checkpoint (relative to project root)
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model_en2vi" / "best_model.pt"
    TOKENIZER_DIR = PROJECT_ROOT / "SentencePiece-from-scratch" / "tokenizer_models"
    
    print("=" * 60)
    print("EVALUATING MODEL ON TEST SET (EN → VI)")
    print("=" * 60)
    
    # Check if checkpoint exists
    if not Path(CHECKPOINT_PATH).exists():
        print(f"\n❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
        print("\nAvailable checkpoints:")
        checkpoints_dir = Path(CHECKPOINT_PATH).parent.parent
        for model_dir in checkpoints_dir.glob('*/'):
            if model_dir.is_dir():
                print(f"  - {model_dir.name}/")
                for ckpt in model_dir.glob('*.pt'):
                    print(f"      {ckpt.name}")
        return
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    config = checkpoint['config']
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Model configuration:")
    print(f"  d_model: {config.d_model}")
    print(f"  layers: {config.n_encoder_layers}")
    print(f"  heads: {config.n_heads}")
    print(f"  Trained epochs: {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    # Initialize data processor
    print("\nInitializing data processor...")
    processor = DataProcessor(Config)
    processor.load_tokenizer(TOKENIZER_DIR)
    
    # Prepare datasets (load test set)
    print("\nPreparing test dataset...")
    datasets = processor.prepare_datasets()
    test_dataset = datasets['test']
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,  # Can use larger batch for evaluation
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor.pad_idx),
        num_workers=0
    )
    
    print(f"  Test samples: {len(test_dataset):,}")
    
    # Create model
    print("\nCreating model...")
    model = BestTransformer(
        src_vocab_size=processor.vocab_size,
        tgt_vocab_size=processor.vocab_size,
        config=config
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    
    print(f"  Model loaded to {config.device}")
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Create evaluator
    evaluator = Evaluator(model, test_loader)
    
    # Show examples
    evaluator.show_examples(processor, num_examples=5)
    
    # Evaluate with beam search
    print("\n\n" + "=" * 60)
    print("EVALUATING WITH BEAM SEARCH")
    print("=" * 60)
    bleu_beam = evaluator.evaluate(use_beam=True, beam_size=4)
    
    # Evaluate with greedy search (faster)
    print("\n\n" + "=" * 60)
    print("EVALUATING WITH GREEDY SEARCH")
    print("=" * 60)
    bleu_greedy = evaluator.evaluate(use_beam=False)
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Beam Search BLEU-4:   {bleu_beam['bleu-4']:.2f}")
    print(f"Greedy Search BLEU-4: {bleu_greedy['bleu-4']:.2f}")
    print(f"Improvement:          +{bleu_beam['bleu-4'] - bleu_greedy['bleu-4']:.2f}")
    print("=" * 60)
    
    # Save results
    results_file = Path(CHECKPOINT_PATH).parent / 'test_results.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("TEST SET EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        f.write(f"Test samples: {len(test_dataset):,}\n\n")
        f.write("Beam Search (beam_size=4):\n")
        f.write(f"  BLEU-1: {bleu_beam['bleu-1']:.2f}\n")
        f.write(f"  BLEU-2: {bleu_beam['bleu-2']:.2f}\n")
        f.write(f"  BLEU-3: {bleu_beam['bleu-3']:.2f}\n")
        f.write(f"  BLEU-4: {bleu_beam['bleu-4']:.2f}\n\n")
        f.write("Greedy Search:\n")
        f.write(f"  BLEU-1: {bleu_greedy['bleu-1']:.2f}\n")
        f.write(f"  BLEU-2: {bleu_greedy['bleu-2']:.2f}\n")
        f.write(f"  BLEU-3: {bleu_greedy['bleu-3']:.2f}\n")
        f.write(f"  BLEU-4: {bleu_greedy['bleu-4']:.2f}\n")
    
    print(f"\n✓ Results saved to: {results_file}")
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
