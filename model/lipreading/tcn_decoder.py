import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Import existing TCN implementation from models folder
from lipreading.models.tcn import TemporalConvNet, MultibranchTemporalConvNet

class TCNDecoder(nn.Module):
    """
    TCN-based decoder for Arabic lip-reading. Uses existing TCN implementation
    from the lipreading/models/tcn.py file.
    """
    def __init__(
        self,
        vocab_size,
        hidden_dim=512,
        num_channels=[384, 384, 384, 384],  # Number of channels in each layer
        kernel_size=3,  # Single kernel size
        dropout=0.2,
        mode='multibranch'  # Whether to use multibranch TCN
    ):
        super(TCNDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Input projection layer to match TCN input dimensions
        self.input_projection = nn.Linear(vocab_size, hidden_dim)
        
        # Use either MultibranchTemporalConvNet or TemporalConvNet depending on mode
        if mode == 'multibranch':
            # For multibranch, we need multiple kernel sizes
            kernel_sizes = [kernel_size, kernel_size*2-1, kernel_size*3-2]  # e.g. [3, 5, 7]
            
            # Create tcn_options dictionary with kernel_size
            tcn_options = {
                'kernel_size': kernel_sizes,
                'dropout': dropout,
                'dwpw': False,  # Depthwise separable convolution
            }
            
            self.tcn = MultibranchTemporalConvNet(
                num_inputs=hidden_dim,
                num_channels=num_channels,
                tcn_options=tcn_options,
                dropout=dropout,
                relu_type='prelu'
            )
        else:
            self.tcn = TemporalConvNet(
                num_inputs=hidden_dim,
                num_channels=num_channels,
                kernel_size=kernel_size,
                dropout=dropout
            )
        
        # Output layer to vocabulary size
        self.output_layer = nn.Linear(num_channels[-1], vocab_size)
        
        # Initialize parameters
        self._reset_parameters()
        
        # For beam search tracking
        self.hidden_dim = hidden_dim
        
    def _reset_parameters(self):
        """
        Initialize the embedding and linear layer parameters.
        """
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, encoder_features, labels=None):
        """
        Forward pass of the TCN decoder.
        Args:
            encoder_features: Tensor of shape [batch_size, seq_len, hidden_size]
            labels: Optional tensor of shape [batch_size, seq_len] for teacher forcing
        Returns:
            logits: Tensor of shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len, hidden_size = encoder_features.size()
        
        # Project input to match TCN input dimensions
        projected_input = self.input_projection(encoder_features)
        
        # Convert to TCN input format [batch, channels, seq_len]
        tcn_input = projected_input.transpose(1, 2)
        
        # Pass through TCN
        tcn_out = self.tcn(tcn_input)  # Output shape: [batch, channels, seq_len]
        
        # Convert back to [batch, seq_len, channels]
        tcn_out = tcn_out.transpose(1, 2)
        
        # Skip connection with encoder features - REMOVED due to dimension mismatch
        # Instead, use only the TCN output directly
        
        # Project to vocabulary size
        logits = self.output_layer(tcn_out)
        
        return logits
    
    def generate(self, encoder_features, maxlen=50):
        """
        Autoregressive generation with the TCN decoder.
        Args:
            encoder_features: Encoder features (batch, seq_len, hidden_dim)
            maxlen: Maximum sequence length
        Returns:
            ys: Generated sequences (batch, seq_len)
            scores: Sequence scores (batch)
        """
        device = encoder_features.device
        batch_size = encoder_features.size(0)
        
        # Initialize with start token (1)
        ys = torch.ones(batch_size, 1).long().to(device)
        scores = torch.zeros(batch_size).to(device)
        
        # Autoregressive generation
        for i in range(maxlen - 1):
            # Forward through decoder
            logits = self.forward(encoder_features)
            
            # Get last token probabilities
            logp = F.log_softmax(logits[:, -1], dim=-1)
            
            # Select next token (greedy decoding)
            next_token = logp.argmax(dim=-1, keepdim=True)
            
            # Update scores
            scores += logp.max(dim=-1)[0]
            
            # Append to generated sequence
            ys = torch.cat([ys, next_token], dim=1)
            
            # Check if all sequences have end tokens (assuming 2 is EOS)
            if (next_token == 2).all():
                break
                
        return ys, scores
    
    def batch_beam_search(self, encoder_features, beam_size=5, maxlen=50):
        """
        Beam search for better sequence generation using CTC-style decoding.
        Args:
            encoder_features: Encoder features (batch, seq_len, hidden_dim)
            beam_size: Number of beams to maintain
            maxlen: Maximum sequence length (not used in this implementation)
        Returns:
            all_results: N-best hypotheses for each item in batch
        """
        device = encoder_features.device
        batch_size = encoder_features.size(0)
        all_results = []
        
        print(f"Starting CTC-style batch beam search with batch_size={batch_size}, beam_size={beam_size}")
        
        # Get logits from encoder features through TCN
        logits = self.forward(encoder_features)  # [batch, seq_len, vocab_size]
        
            
        log_probs = F.log_softmax(logits, dim=2)  # Apply log_softmax
        
        # Process each batch item separately
        for b in range(batch_size):
            print(f"\nBeam search for batch item {b+1}/{batch_size}")
            
            # Get log probs for this batch item
            batch_log_probs = log_probs[b].cpu()  # [seq_len, vocab_size]
            seq_len, vocab_size = batch_log_probs.shape
            blank_index = 0  # Index of blank token
            
            # Initialize beam with just the blank token (CTC style)
            beam = [{"score": 0.0, "yseq": []}]
            
            # Process each timestep
            for t in range(seq_len):
                logp_t = batch_log_probs[t]  # (vocab_size,)
                new_beam = []
                
                # For each hypothesis in the beam
                for hyp in beam:
                    # Current token sequence and score
                    y_sequence = hyp['yseq']
                    base_score = hyp['score']
                    
                    # Option 1: Add blank (don't emit a new token)
                    blank_score = base_score + float(logp_t[blank_index].item())
                    new_beam.append({'yseq': y_sequence.copy(), 'score': blank_score})
                    
                    # Option 2: Emit each possible token
                    for c in range(1, vocab_size):  # Skip blank
                        # Skip if probability is too low (avoid garbage predictions)
                        if logp_t[c].item() < -10:  # Skip very low probability tokens
                            continue
                            
                        # Skip if we're repeating the most recent token (CTC rules)
                        if y_sequence and y_sequence[-1] == c:
                            continue
                            
                        # Add token to new sequence
                        new_y = y_sequence.copy()
                        new_y.append(c)
                        new_score = base_score + float(logp_t[c].item())
                        new_beam.append({'yseq': new_y, 'score': new_score})
                
                # Keep top beam_size hypotheses
                beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)[:beam_size]
                
                # Print progress every 10 steps
                if (t + 1) % 10 == 0:
                    print(f"  Step {t+1}/{seq_len} - Top beam: {beam[0]['yseq']} (score: {beam[0]['score']:.4f})")
                    
                    # Print token probabilities for visualization
                    if t < 5:  # Only for first few steps to avoid clutter
                        top_tokens = torch.topk(logp_t, 5)
                        print(f"    Top tokens at step {t+1}: {top_tokens.indices.tolist()} with log probs: {top_tokens.values.tolist()}")
            
            # Final processing - add SOS/EOS tokens for compatibility with the rest of the code
            final_beam = []
            for hyp in beam:
                # Add SOS (1) and EOS (2) tokens
                final_seq = [1] + hyp['yseq'] + [2] if hyp['yseq'] else [1, 2]
                final_beam.append({"yseq": final_seq, "score": hyp['score']})
            
            # Sort final beam by score
            final_beam = sorted(final_beam, key=lambda x: x['score'], reverse=True)
            
            # Filter out beams with no actual content (just SOS/EOS tokens)
            meaningful_beams = [b for b in final_beam if len(b['yseq']) > 2]
            if meaningful_beams:
                final_beam = meaningful_beams
            
            # Print top results
            print(f"  Final results for batch {b+1}:")
            for i, hyp in enumerate(final_beam[:3]):  # Print top 3
                print(f"    Beam {i+1}: {hyp['yseq']} (score: {hyp['score']:.4f})")
            
            # Add to results
            all_results.append(final_beam)
            
        return all_results 