"""
A4: Language Modeling with Transformers
LING-L 665, Spring 2025

Student: Mitanshu Bhoot
"""

import os
import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ShakespeareDataset(Dataset):
    """Dataset for Shakespeare text."""

    def __init__(self, file_path: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Read the text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize the entire text
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        # Return number of possible sequences
        return max(1, len(self.tokens) - self.max_length)

    def __getitem__(self, idx):
        # Get a sequence of max_length + 1 tokens (input + target)
        sequence = self.tokens[idx:idx + self.max_length + 1]

        # Pad if necessary
        if len(sequence) < self.max_length + 1:
            sequence = sequence + [self.tokenizer.eot_token] * (self.max_length + 1 - len(sequence))

        # Split into input and target
        x = torch.tensor(sequence[:-1], dtype=torch.long)
        y = torch.tensor(sequence[1:], dtype=torch.long)

        return x, y


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, return_attention_weights=False):
        """
        Forward pass for multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            return_attention_weights: If True, also return attention weights

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
            If return_attention_weights=True, also returns attention weights of shape
            (batch_size, num_heads, seq_len, seq_len)
        """
        # 1. Get batch_size and seq_len from x.shape
        batch_size, seq_len, _ = x.shape
        
        # 2. Project x to Q, K, V using self.W_q, self.W_k, self.W_v
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)  # (batch_size, seq_len, d_model)
        V = self.W_v(x)  # (batch_size, seq_len, d_model)
        
        # 3. Reshape Q, K, V to split into multiple heads:
        #    - Shape should be (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 4. Compute attention scores: scores = Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        
        # 5. Create causal mask to prevent attending to future positions
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(scores.device)
        
        # 6. Apply mask by setting masked positions to -inf
        scores = scores.masked_fill(mask, float('-inf'))
        
        # 7. Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # 8. Compute weighted sum: output = attention_weights @ V
        output = torch.matmul(attention_weights, V)
        # output shape: (batch_size, num_heads, seq_len, d_k)
        
        # 9. Reshape output back to (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 10. Apply output projection: self.W_o(output)
        output = self.W_o(output)
        
        # If return_attention_weights is True, return (output, attention_weights)
        if return_attention_weights:
            return output, attention_weights
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network (two linear layers with ReLU)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    """A single Transformer block with attention and feed-forward layers."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, use_pre_norm: bool = True):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.use_pre_norm = use_pre_norm

    def forward(self, x, return_attention_weights=False):
        """
        Forward pass for the Transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            return_attention_weights: If True, also return attention weights

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
            If return_attention_weights=True, also returns attention weights
        """
        if self.use_pre_norm:
            # Pre-norm architecture:
            # x = x + attention(layer_norm(x))
            # x = x + ffn(layer_norm(x))
            
            # Attention sublayer with residual connection
            if return_attention_weights:
                attn_output, attn_weights = self.attention(self.ln1(x), return_attention_weights=True)
                x = x + attn_output
            else:
                x = x + self.attention(self.ln1(x))
            
            # Feed-forward sublayer with residual connection
            x = x + self.ffn(self.ln2(x))
            
        else:
            # Post-norm architecture:
            # x = layer_norm(x + attention(x))
            # x = layer_norm(x + ffn(x))
            
            # Attention sublayer with residual connection
            if return_attention_weights:
                attn_output, attn_weights = self.attention(x, return_attention_weights=True)
                x = self.ln1(x + attn_output)
            else:
                x = self.ln1(x + self.attention(x))
            
            # Feed-forward sublayer with residual connection
            x = self.ln2(x + self.ffn(x))
        
        # Return output and attention weights if requested
        if return_attention_weights:
            return x, attn_weights
        
        return x


class TransformerLM(nn.Module):
    """Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_length: int,
        use_pre_norm: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings (learned)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, use_pre_norm)
            for _ in range(num_layers)
        ])

        # Final layer norm (always applied in both pre-norm and post-norm)
        self.ln_f = nn.LayerNorm(d_model)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, return_attention=False):
        """
        Forward pass for the Transformer language model.

        Args:
            x: Input token indices of shape (batch_size, seq_len)
            return_attention: If True, also return attention weights from all layers

        Returns:
            logits: Logits of shape (batch_size, seq_len, vocab_size)
            If return_attention=True, also returns:
                attention_weights: List of attention weight tensors, one per layer
                                  Each has shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len = x.shape

        # Get token embeddings
        token_emb = self.token_embedding(x)  # (batch_size, seq_len, d_model)

        # Get positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        pos_emb = self.position_embedding(positions)  # (1, seq_len, d_model)

        # Combine embeddings
        x = token_emb + pos_emb

        # Pass through Transformer blocks
        if return_attention:
            attention_weights = []
            for block in self.blocks:
                x, attn = block(x, return_attention_weights=True)
                attention_weights.append(attn)
        else:
            for block in self.blocks:
                x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        if return_attention:
            return logits, attention_weights
        return logits


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits = model(x)

        # Compute loss (reshape for cross entropy)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Track loss
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / total_tokens
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {avg_loss:.4f}")

    return total_loss / total_tokens


def evaluate(model, dataloader, device):
    """Evaluate the model and return average loss."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    return total_loss / total_tokens


def compute_perplexity(loss):
    """Compute perplexity from loss."""
    return math.exp(loss)


def generate(model, tokenizer, prompt: str, max_length: int = 100, temperature: float = 1.0, device='cpu'):
    """
    Generate text from the model.

    Args:
        model: The trained TransformerLM
        tokenizer: The tiktoken tokenizer
        prompt: The prompt text to start generation
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        device: Device to run generation on

    Returns:
        Generated text as a string
    """
    model.eval()

    # Encode the prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions for the last token
            # (only use up to max_seq_length tokens as context)
            context = tokens[:, -model.max_seq_length:]
            logits = model(context)
            logits = logits[:, -1, :] / temperature

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to the sequence
            tokens = torch.cat([tokens, next_token], dim=1)

            # Stop if we generate an end-of-text token
            if next_token.item() == tokenizer.eot_token:
                break

    # Decode and return
    generated_tokens = tokens.squeeze(0).tolist()
    return tokenizer.decode(generated_tokens)


def visualize_attention(model, tokenizer, text: str, epoch: int, device='cpu', output_dir='attention'):
    """
    Visualize attention weights for a given text and save heatmaps.

    Args:
        model: The trained TransformerLM
        tokenizer: The tiktoken tokenizer
        text: The text to visualize attention for
        epoch: Current epoch number (for filename)
        device: Device to run on
        output_dir: Directory to save attention heatmaps
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Tokenize the text
    tokens = tokenizer.encode(text)
    token_strs = [tokenizer.decode([t]) for t in tokens]

    # Convert to tensor
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Get attention weights
    with torch.no_grad():
        _, attention_weights = model(x, return_attention=True)

    # attention_weights is a list of tensors, one per layer
    # Each has shape (batch_size, num_heads, seq_len, seq_len)

    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]

    # Save heatmap for each layer and head
    for layer_idx in range(num_layers):
        attn = attention_weights[layer_idx].squeeze(0)  # (num_heads, seq_len, seq_len)

        for head_idx in range(num_heads):
            head_attn = attn[head_idx].cpu().numpy()  # (seq_len, seq_len)

            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(head_attn, cmap='viridis', aspect='auto')

            # Set ticks and labels
            ax.set_xticks(range(len(token_strs)))
            ax.set_yticks(range(len(token_strs)))
            ax.set_xticklabels(token_strs, rotation=90, fontsize=8)
            ax.set_yticklabels(token_strs, fontsize=8)

            # Add colorbar
            plt.colorbar(im, ax=ax)

            # Add title
            ax.set_title(f'Layer {layer_idx + 1}, Head {head_idx + 1}, Epoch {epoch}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')

            # Save figure
            filename = f'layer-{layer_idx + 1}_head-{head_idx + 1}_epoch-{epoch}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()

    print(f"Saved {num_layers * num_heads} attention heatmaps to {output_dir}/")


def plot_training_curve(losses: List[float], save_path: str):
    """Plot and save the training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Training curve saved to {save_path}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--use_pre_norm', action='store_true', default=False)
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    print(f"Vocabulary size: {vocab_size}")

    # Load datasets
    print("Loading datasets...")
    train_dataset = ShakespeareDataset(
        os.path.join(args.data_dir, 'train.txt'),
        tokenizer,
        args.max_length
    )
    dev_dataset = ShakespeareDataset(
        os.path.join(args.data_dir, 'dev.txt'),
        tokenizer,
        args.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train size: {len(train_dataset)} sequences")
    print(f"Dev size: {len(dev_dataset)} sequences")

    # Show sample tokenization
    print("\nSample tokenization:")
    with open(os.path.join(args.data_dir, 'train.txt'), 'r', encoding='utf-8') as f:
        sample_text = f.readline().strip()
    sample_tokens = tokenizer.encode(sample_text)
    print(f"Text: {sample_text[:100]}...")
    print(f"Tokens: {sample_tokens[:20]}...")
    print(f"Decoded: {tokenizer.decode(sample_tokens[:20])}...")
    print()

    # Initialize model
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_length=args.max_length,
        use_pre_norm=args.use_pre_norm
    ).to(device)
    print("Model initialized.")
    print(model)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Architecture: {'Pre-norm' if args.use_pre_norm else 'Post-norm'}")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Training loop
    print("\nTraining...")
    train_losses = []
    best_dev_loss = float('inf')

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        train_ppl = compute_perplexity(train_loss)

        # Evaluate
        dev_loss = evaluate(model, dev_loader, device)
        dev_ppl = compute_perplexity(dev_loss)

        print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
        print(f"Dev Loss: {dev_loss:.4f}, Dev PPL: {dev_ppl:.2f}")

        # Generate sample texts
        print("\nSample generations:")
        print("=" * 80)
        sample_prompts = ["ROMEO:", "JULIET:", "To be or not to be"]
        for i, prompt in enumerate(sample_prompts, 1):
            generated = generate(model, tokenizer, prompt, max_length=50, temperature=0.8, device=device)
            print(f"{i}. {generated}")
            print()
        print("=" * 80)

        # Visualize attention for sample text
        sample_text = "that which we call a rose By any other name would smell as sweet"
        print(f"\nVisualizing attention for: '{sample_text}'")
        visualize_attention(model, tokenizer, sample_text, epoch + 1, device=device)

        # Save best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved best model!")

    # Plot training curve
    norm_type = "prenorm" if args.use_pre_norm else "postnorm"
    plot_training_curve(train_losses, f'training_curve_{norm_type}.png')

    # Load best model for generation
    model.load_state_dict(torch.load('best_model.pt'))

    # Generate some text
    print("\n" + "="*80)
    print("Generated text:")
    print("="*80)
    prompt = "ROMEO:"
    generated = generate(model, tokenizer, prompt, max_length=100, temperature=0.8, device=device)
    print(generated)
    print("="*80)

    # Final evaluation
    print(f"\nFinal Dev Perplexity: {compute_perplexity(best_dev_loss):.2f}")
    print(f"Baseline Perplexity (uniform distribution): {vocab_size}")


if __name__ == '__main__':
    main()
