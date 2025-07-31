import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
from typing import Optional, Tuple

import geoopt 
from geoopt import ManifoldParameter
from geoopt import PoincareBall


class HyperbolicLinear(nn.Module):
        def __init__(self, in_features, out_features, manifold, c):
            super().__init__()
            self.weight = ManifoldParameter(torch.randn(out_features, in_features) * 0.01, manifold=manifold)
            self.bias = ManifoldParameter(torch.zeros(out_features), manifold=manifold)
            self.manifold = manifold
            self.c = c

        def forward(self, x):
            x = self.manifold.mobius_matvec(self.weight, x)
            x = self.manifold.mobius_add(x, self.bias)
            return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network"""
    
    def __init__(self, d_model, d_ff, manifold, c, dropout = 0.1):
        super().__init__()
        self.manifold = manifold
        self.c =c
        if self.c != 0:
            self.w_1 = HyperbolicLinear(d_model, d_ff, self.manifold, self.c)
            self.w_2 = HyperbolicLinear(d_ff, d_model, self.manifold, self.c)
        else:
            self.w_1 = nn.Linear(d_model, d_ff)
            self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        if self.c != 0:
            x = self.manifold.expmap0(x)
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        if self.c != 0:
            x = self.manifold.logmap0(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class EncoderLayer(nn.Module):
    """Single Encoder Layer"""
    
    def __init__(self, d_model, n_heads, d_ff, manifold, c, dropout = 0.1):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, self.manifold, self.c, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        if self.c != 0:
            x = self.manifold.expmap0(x)
            attn_output = self.manifold.expmap0(attn_output)
            x = self.manifold.mobius_add(x, attn_output)
            x = self.manifold.logmap0(x)
        else:
            x = x + attn_output
        x = self.norm1(x)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        if self.c != 0:
            ff_output = self.manifold.expmap0(ff_output)
            x = self.manifold.expmap0(x)
            x = self.manifold.mobius_add(x, ff_output)
            x = self.manifold.logmap0(x)
        else:
            x = x + ff_output
        x = self.norm2(x)
        
        return x


class DecoderLayer(nn.Module):
    """Single Decoder Layer"""
    
    def __init__(self, d_model, n_heads, d_ff, manifold, c, dropout: float = 0.1):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, self.manifold, self.c, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention (masked)
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        self_attn_output = self.dropout(self_attn_output)
        if self.c != 0:
            x = self.manifold.expmap0(x)
            self_attn_output = self.manifold.expmap0(self_attn_output)
            x = self.manifold.mobius_add(x, self_attn_output)
            x = self.manifold.logmap0(x)
        else:
            x = x + self_attn_output
        x = self.norm1(x)
        
        # Cross-attention
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        cross_attn_output = self.dropout(cross_attn_output)
        if self.c != 0:
            x = self.manifold.expmap0(x)
            cross_attn_output = self.manifold.expmap0(cross_attn_output)
            x = self.manifold.mobius_add(x, cross_attn_output)
            x = self.manifold.logmap0(x)
        else:
            x = x + cross_attn_output
        x = self.norm2(x)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        if self.c != 0:
            x = self.manifold.expmap0(x)
            ff_output = self.manifold.expmap0(ff_output)
            x = self.manifold.mobius_add(x, ff_output)
            x = self.manifold.logmap0(x)
        else:
            x = x + ff_output
        x = self.norm3(x)
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, 
                 n_layers: int, d_ff: int, max_seq_length: int, manifold, c, dropout: float = 0.1):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, self.manifold, self.c, dropout) 
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        # Embedding + positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
            
        return x


class TransformerDecoder(nn.Module):
    """Transformer Decoder"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, 
                 n_layers: int, d_ff: int, max_seq_length: int, manifold, c, dropout: float = 0.1):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, self.manifold, self.c, dropout) 
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        # Embedding + positional encoding
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
            
        return x


class Transformer(nn.Module):
    """Complete Transformer Model with Encoder-Decoder Architecture"""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512, 
                 n_heads: int = 8, n_layers: int = 6, d_ff: int = 2048, 
                 max_seq_length: int = 5000, c=1.0, dropout: float = 0.1):
        super().__init__()

        self.c = c
        self.manifold = geoopt.PoincareBall(c=self.c)
        
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length, self.manifold, self.c, dropout
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length, self.manifold, self.c, dropout
        )
        self.output_projection = nn.Linear(d_model, 1)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass
        Args:
            src: Source sequence [batch_size, src_seq_len]
            tgt: Target sequence [batch_size, tgt_seq_len]
            src_mask: Source mask [batch_size, 1, 1, src_seq_len]
            tgt_mask: Target mask [batch_size, 1, tgt_seq_len, tgt_seq_len]
        """
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        decoder_output = decoder_output[:, 0, :] 
        output = self.output_projection(decoder_output)
        return output
    
    def encode(self, src, src_mask=None):
        """Encode source sequence"""
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """Decode target sequence"""
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_projection(decoder_output)


def create_padding_mask(seq, pad_idx=0):
    """Create padding mask"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size):
    """Create look-ahead mask for decoder"""
    mask = torch.triu(torch.ones((size, size)), diagonal=1)
    return mask == 0


def create_masks(src, tgt, pad_idx=0):
    """Create all necessary masks"""
    src_mask = create_padding_mask(src, pad_idx)
    
    tgt_mask = create_padding_mask(tgt, pad_idx)
    look_ahead_mask = create_look_ahead_mask(tgt.size(1))
    tgt_mask = tgt_mask & look_ahead_mask.unsqueeze(0)
    
    return src_mask, tgt_mask


# class TransformerConfig:
#     """Configuration class for Transformer"""
    
#     def __init__(self, **kwargs):
#         # Model architecture
#         self.src_vocab_size = kwargs.get('src_vocab_size', 30000)
#         self.tgt_vocab_size = kwargs.get('tgt_vocab_size', 30000)
#         self.d_model = kwargs.get('d_model', 512)
#         self.n_heads = kwargs.get('n_heads', 8)
#         self.n_layers = kwargs.get('n_layers', 6)
#         self.d_ff = kwargs.get('d_ff', 2048)
#         self.max_seq_length = kwargs.get('max_seq_length', 5000)
#         self.dropout = kwargs.get('dropout', 0.1)
        
#         # Training
#         self.learning_rate = kwargs.get('learning_rate', 0.0001)
#         self.batch_size = kwargs.get('batch_size', 32)
#         self.epochs = kwargs.get('epochs', 10)
#         self.warmup_steps = kwargs.get('warmup_steps', 4000)
        
#         # Data
#         self.pad_idx = kwargs.get('pad_idx', 0)
#         self.sos_idx = kwargs.get('sos_idx', 1)
#         self.eos_idx = kwargs.get('eos_idx', 2)


# def get_args():
#     """Parse command line arguments"""
#     parser = argparse.ArgumentParser(description='Transformer Model')
    
#     # Model parameters
#     parser.add_argument('--src_vocab_size', type=int, default=30000)
#     parser.add_argument('--tgt_vocab_size', type=int, default=30000)
#     parser.add_argument('--d_model', type=int, default=512)
#     parser.add_argument('--n_heads', type=int, default=8)
#     parser.add_argument('--n_layers', type=int, default=6)
#     parser.add_argument('--d_ff', type=int, default=2048)
#     parser.add_argument('--max_seq_length', type=int, default=5000)
#     parser.add_argument('--dropout', type=float, default=0.1)
    
#     # Training parameters
#     parser.add_argument('--learning_rate', type=float, default=0.0001)
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--epochs', type=int, default=10)
#     parser.add_argument('--warmup_steps', type=int, default=4000)
    
#     # Data parameters
#     parser.add_argument('--pad_idx', type=int, default=0)
#     parser.add_argument('--sos_idx', type=int, default=1)
#     parser.add_argument('--eos_idx', type=int, default=2)
    
#     return parser.parse_args()


# def example_usage():
#     """Example of how to use the Transformer model"""
    
#     # Configuration
#     config = TransformerConfig(
#         src_vocab_size=10000,
#         tgt_vocab_size=10000,
#         d_model=512,
#         n_heads=8,
#         n_layers=6,
#         d_ff=2048,
#         max_seq_length=100,
#         dropout=0.1
#     )
    
#     # Create model
#     model = Transformer(
#         src_vocab_size=config.src_vocab_size,
#         tgt_vocab_size=config.tgt_vocab_size,
#         d_model=config.d_model,
#         n_heads=config.n_heads,
#         n_layers=config.n_layers,
#         d_ff=config.d_ff,
#         max_seq_length=config.max_seq_length,
#         dropout=config.dropout
#     )
    
#     # Example input (batch_size=2, seq_len=10)
#     src = torch.randint(1, 1000, (2, 10))
#     tgt = torch.randint(1, 1000, (2, 8))
    
#     # Create masks
#     src_mask, tgt_mask = create_masks(src, tgt, pad_idx=0)
    
#     # Forward pass
#     output = model(src, tgt, src_mask, tgt_mask)
#     print(f"Output shape: {output.shape}")  # [batch_size, tgt_seq_len, tgt_vocab_size]
    
#     # Calculate model parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")


# if __name__ == "__main__":
#     # Parse arguments
#     args = get_args()
    
#     # Create config from args
#     config = TransformerConfig(**vars(args))
    
#     # Run example
#     print("Running Transformer example...")
#     example_usage()
    
#     print("\nTransformer model implementation completed!")
#     print("Key features:")
#     print("- Modular design with separate encoder/decoder")
#     print("- Configurable architecture parameters")
#     print("- Compatible with any dataset (through vocab_size parameters)")
#     print("- Includes masking utilities")
#     print("- Command-line argument support")
#     print("- Proper parameter initialization")