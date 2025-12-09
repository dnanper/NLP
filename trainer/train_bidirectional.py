"""
Bidirectional Training Script for Vietnamese ↔ English Translation

Train a SINGLE model that can translate in BOTH directions:
- Vietnamese → English
- English → Vietnamese

Strategy: Add language direction token to indicate target language
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models_best import BestTransformer, TransformerConfig, LabelSmoothingLoss
from utils.data_processing import DataProcessor, collate_fn
from config import Config


class BidirectionalDataset(Dataset):
    """
    Dataset that provides both translation directions
    Each (vi, en) pair is used twice:
    - Once as vi → en
    - Once as en → vi
    """
    
    def __init__(self, src_data: List[List[int]], tgt_data: List[List[int]]):
        """
        Args:
            src_data: English sentences (token IDs)
            tgt_data: Vietnamese sentences (token IDs)
        """
        assert len(src_data) == len(tgt_data)
        self.en_data = src_data  # Assuming src is English
        self.vi_data = tgt_data  # Assuming tgt is Vietnamese
    
    def __len__(self):
        # Each pair used twice (both directions)
        return len(self.en_data) * 2
    
    def __getitem__(self, idx):
        # Get actual pair index
        pair_idx = idx // 2
        direction = idx % 2  # 0 = vi→en, 1 = en→vi
        
        if direction == 0:
            # Vietnamese → English
            return {
                'src': torch.tensor(self.vi_data[pair_idx], dtype=torch.long),
                'tgt': torch.tensor(self.en_data[pair_idx], dtype=torch.long)
            }
        else:
            # English → Vietnamese
            return {
                'src': torch.tensor(self.en_data[pair_idx], dtype=torch.long),
                'tgt': torch.tensor(self.vi_data[pair_idx], dtype=torch.long)
            }


class BidirectionalTrainer:
    """
    Trainer for bidirectional translation model
    """
    
    def __init__(self, model: BestTransformer, config: TransformerConfig,
                 train_loader: DataLoader, val_loader: DataLoader,
                 save_dir: str):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = LabelSmoothingLoss(
            num_classes=model.tgt_vocab_size,
            smoothing=config.label_smoothing,
            ignore_index=config.pad_idx
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                return (self.config.warmup_steps / step) ** 0.5
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            src = batch['src'].to(self.config.device)
            tgt = batch['tgt'].to(self.config.device)
            src_mask = batch['src_mask'].to(self.config.device)
            tgt_mask = batch['tgt_mask'].to(self.config.device)
            
            # Prepare input and output
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = tgt_mask[:, :, :-1, :-1]
            
            # Forward pass
            logits = self.model(src, tgt_input, src_mask, tgt_mask)
            
            # Compute loss
            loss = self.criterion(logits, tgt_output)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
            
            if self.global_step % 100 == 0:
                self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        pbar = tqdm(self.val_loader, desc="Validating")
        
        for batch in pbar:
            src = batch['src'].to(self.config.device)
            tgt = batch['tgt'].to(self.config.device)
            src_mask = batch['src_mask'].to(self.config.device)
            tgt_mask = batch['tgt_mask'].to(self.config.device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = tgt_mask[:, :, :-1, :-1]
            
            # Forward pass
            logits = self.model(src, tgt_input, src_mask, tgt_mask)
            
            # Compute loss
            loss = self.criterion(logits, tgt_output)
            total_loss += loss.item()
            
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, filename: str = 'checkpoint.pt'):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history
        }
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"✓ Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"✓ Checkpoint loaded from {checkpoint_path}")
        print(f"  Resuming from epoch {self.epoch}, step {self.global_step}")
    
    def train(self, num_epochs: int, save_every: int = 1, resume_from: Optional[str] = None):
        """Train the model"""
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        
        print("=" * 60)
        print(f"Starting BIDIRECTIONAL training for {num_epochs} epochs")
        print(f"Device: {self.config.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print("=" * 60)
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch + 1
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Track metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Compute epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Epoch {self.epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {self.scheduler.get_last_lr()[0]:.6f}")
            print(f"  Time:       {epoch_time:.1f}s")
            print(f"{'='*60}\n")
            
            # Save checkpoint
            if self.epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.epoch}.pt')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
                print(f"✓ New best model saved! Val Loss: {val_loss:.4f}\n")
            
            # Save latest checkpoint
            self.save_checkpoint('latest.pt')
        
        # Final save
        self.save_checkpoint('final_model.pt')
        
        # Plot training curves
        self.plot_training_curves()
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total steps: {self.global_step}")
        print("=" * 60)
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(epochs, self.history['val_loss'], label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate
        steps = range(len(self.history['learning_rate']))
        ax2.plot(steps, self.history['learning_rate'])
        ax2.set_xlabel('Step (x100)')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150)
        print(f"✓ Training curves saved to {self.save_dir / 'training_curves.png'}")


def main():
    """Main training function for bidirectional model"""
    
    # ========== Configuration ==========
    
    # Get project root directory (parent of trainer/)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Paths (relative to project root)
    SAVE_DIR = PROJECT_ROOT / "checkpoints" / "best_model_bidirectional"
    TOKENIZER_DIR = PROJECT_ROOT / "SentencePiece-from-scratch" / "tokenizer_models"
    
    # Model configuration - THAY ĐỔI KÍCH CỠ MODEL TẠI ĐÂY:
    # .small() - 256d, 4 layers, ~60M params (fast training)
    # .base()  - 512d, 6 layers, ~65M params (balanced) ⭐ RECOMMENDED
    # .large() - 1024d, 6 layers, ~213M params (best quality)
    # .deep()  - 512d, 12 layers (very deep)
    config = TransformerConfig.base()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.max_len = Config.MAX_LEN
    config.dropout = 0.1
    config.label_smoothing = 0.1
    config.warmup_steps = 8000
    config.learning_rate = 1e-4
    config.grad_clip = 1.0
    
    # Training configuration
    BATCH_SIZE = Config.BATCH_SIZE
    NUM_EPOCHS = 30
    
    print("=" * 60)
    print("BIDIRECTIONAL Vietnamese ↔ English Translation Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Model: {config.d_model}d, {config.n_encoder_layers} layers")
    print(f"  Direction: Vietnamese ↔ English (BOTH)")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max length: {config.max_len}")
    print(f"  Device: {config.device}")
    print("=" * 60)
    
    # ========== Initialize Data Processor ==========
    
    print("\nInitializing data processor...")
    processor = DataProcessor(Config)
    processor.load_tokenizer(TOKENIZER_DIR)
    
    VOCAB_SIZE = processor.vocab_size
    print(f"  Vocabulary size: {VOCAB_SIZE:,}")
    
    # ========== Prepare Bidirectional Dataset ==========
    
    print("\nPreparing bidirectional datasets...")
    datasets = processor.prepare_datasets()  # Returns {'train', 'validation', 'test'}
    
    # Create bidirectional datasets for train and validation
    train_bidir = BidirectionalDataset(datasets['train'].src_data, datasets['train'].tgt_data)
    val_bidir = BidirectionalDataset(datasets['validation'].src_data, datasets['validation'].tgt_data)
    
    print(f"  Train pairs: {len(datasets['train']):,} → Bidirectional: {len(train_bidir):,} (2x)")
    print(f"  Validation pairs: {len(datasets['validation']):,} → Bidirectional: {len(val_bidir):,} (2x)")
    print(f"  Test pairs: {len(datasets['test']):,}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_bidir,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor.pad_idx),
        num_workers=0,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_bidir,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor.pad_idx),
        num_workers=0,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    print(f"  Training batches: {len(train_loader):,}")
    print(f"  Validation batches: {len(val_loader):,}")
    
    # ========== Create Model ==========
    
    print("\nCreating bidirectional model...")
    config.pad_idx = processor.pad_idx
    config.bos_idx = processor.sos_idx
    config.eos_idx = processor.eos_idx
    
    model = BestTransformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        config=config
    )
    
    # ========== Create Trainer ==========
    
    trainer = BidirectionalTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=SAVE_DIR
    )
    
    # ========== Train ==========
    
    # To resume: trainer.train(NUM_EPOCHS, save_every=1, resume_from=os.path.join(SAVE_DIR, 'latest.pt'))
    trainer.train(NUM_EPOCHS, save_every=1)
    
    print("\n✓ Bidirectional training complete!")
    print(f"  Model can now translate BOTH directions:")
    print(f"    - Vietnamese → English")
    print(f"    - English → Vietnamese")
    print(f"  Best model saved to: {SAVE_DIR}/best_model.pt")


if __name__ == '__main__':
    main()
