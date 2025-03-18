import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Import from espnet
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.decoder.transformer_decoder import DecoderLayer

class ArabicTransformerDecoder(nn.Module):
    """
    Transformer Decoder for Arabic lip reading system.
    """
    def __init__(
        self,
        vocab_size,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        self_attention_dropout_rate=0.1,
        src_attention_dropout_rate=0.1,
        normalize_before=True,
        concat_after=False
    ):
        super(ArabicTransformerDecoder, self).__init__()
        
        # Embedding layer converts token indices to vectors and adds positional encoding
        self.embedding = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, attention_dim),
            PositionalEncoding(attention_dim, positional_dropout_rate)
        )
        
        # Create decoder blocks
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        
        # Add layer norm if using pre-norm transformer
        if normalize_before:
            self.after_norm = LayerNorm(attention_dim)
            
        # Output linear layer
        self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        
        # Initialize parameters with xavier uniform
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward_one_step(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):
        """Forward one step in generation.
        
        Args:
            tgt: Input token ids, int64 (batch, maxlen_out)
            tgt_mask: Input token mask, (batch, maxlen_out)
            memory: Encoded memory, float32 (batch, maxlen_in, feat)
            memory_mask: Encoded memory mask, (batch, maxlen_in)
            cache: Cached output list of (batch, max_time_out-1, size)
            
        Returns:
            y: Output tensor (batch, maxlen_out, vocab_size)
            cache: List of cached outputs [(batch, max_time_out-1, size)]
        """
        # Convert target tokens to embeddings
        x = self.embedding(tgt)
        
        # Create new cache or use provided cache
        new_cache = []
        if cache is None:
            cache = [None] * len(self.decoders)
        
        # Forward through each decoder layer
        for i, (decoder, cache_layer) in enumerate(zip(self.decoders, cache)):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, memory_mask, cache=cache_layer
            )
            new_cache.append(x)
        
        # Apply layer norm if needed
        if self.after_norm is not None:
            x = self.after_norm(x)
            
        # Get final output probabilities
        y = self.output_layer(x)
        
        return y, new_cache
    
    def forward(self, tgt, tgt_mask, memory, memory_mask=None):
        """Forward decoder.
        
        Args:
            tgt: Input token ids, int64 (batch, maxlen_out)
            tgt_mask: Input token mask, (batch, maxlen_out)
            memory: Encoded memory, float32 (batch, maxlen_in, feat)
            memory_mask: Encoded memory mask, (batch, maxlen_in)
            
        Returns:
            x: Decoded token scores (batch, maxlen_out, vocab_size)
        """
        x = self.embedding(tgt)
        
        # Forward through decoder layers
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(
                x, tgt_mask, memory, memory_mask
            )
        
        # Apply layer norm if needed
        if self.after_norm is not None:
            x = self.after_norm(x)
            
        # Get final output logits
        x = self.output_layer(x)
        
        return x
    
    def generate(self, memory, memory_mask, maxlen=50):
        """Generate sequence using teacher-forcing.
        
        Args:
            memory: Encoded memory, float32 (batch, maxlen_in, feat)
            memory_mask: Encoded memory mask, (batch, maxlen_in)
            maxlen: Maximum output length
            
        Returns:
            ys: Generated sequences
            scores: Sequence scores
        """
        device = memory.device
        batch_size = memory.size(0)
        
        # Initialize with BOS token (assuming 0 is blank/padding, so use 1 as BOS)
        ys = torch.ones(batch_size, 1).long().to(device) * 1  # Start with <sos> token
        scores = torch.zeros(batch_size).to(device)
        
        # Create cache for fast decoding
        cache = None
        
        # Generate tokens one by one
        for i in range(maxlen):
            # Create mask for the current output sequence
            tgt_mask = subsequent_mask(ys.size(-1)).unsqueeze(0).to(device)
            
            # Forward one step
            logp, cache = self.forward_one_step(
                ys, tgt_mask, memory, memory_mask, cache=cache
            )
            
            # Pick highest probability for next token
            logp = logp[:, -1]  # Last time step
            prob = F.log_softmax(logp, dim=-1)
            
            # Get next token
            next_token = prob.argmax(dim=-1).unsqueeze(-1)
            
            # Add next token to sequence
            ys = torch.cat([ys, next_token], dim=-1)
            
            # Add to scores
            scores += prob.max(dim=-1)[0]
            
            # Check if all sequences reached EOS (assuming 2 is EOS)
            if (next_token == 2).all():
                break
        
        return ys, scores
    
    def batch_beam_search(self, memory, memory_mask, beam_size=5, maxlen=50):
        """
        Perform batch beam search on the decoder.
        
        Args:
            memory: Encoder output (batch, maxlen_in, feat)
            memory_mask: Encoder mask (batch, maxlen_in)
            beam_size: Beam size for search
            maxlen: Maximum output length
            
        Returns:
            nbest_hyps: N-best hypothesis list of lists
        """
        device = memory.device
        batch_size = memory.size(0)
        
        all_results = []
        
        # Process each batch item separately
        for b in range(batch_size):
            # Extract single example from batch
            single_memory = memory[b].unsqueeze(0)  # (1, maxlen_in, feat)
            if memory_mask is not None:
                single_memory_mask = memory_mask[b].unsqueeze(0)  # (1, maxlen_in)
            else:
                single_memory_mask = None
            
            # Initialize beam with start-of-sequence token
            # Using index 1 as the BOS token (0 is blank/pad)
            beams = [
                {"score": 0.0, "yseq": [1], "cache": None}
            ]
            
            for step in range(maxlen):
                # All beams have reached end
                if len(beams) == 0:
                    break
                
                # Prepare input for current step
                beam_size_now = len(beams)
                
                # Create token tensors for all beams
                token_ids = [b["yseq"] for b in beams]
                ylen = max(len(y) for y in token_ids)
                ys = [np.pad(y, (0, ylen - len(y)), "constant") for y in token_ids]
                ys = torch.tensor(ys, dtype=torch.long, device=device)
                
                # Create mask
                tgt_mask = subsequent_mask(ylen).unsqueeze(0).to(device)
                tgt_mask = tgt_mask.repeat(beam_size_now, 1, 1)
                
                # Expand encoder output to match beam size
                beam_memory = single_memory.repeat(beam_size_now, 1, 1)
                if single_memory_mask is not None:
                    beam_memory_mask = single_memory_mask.repeat(beam_size_now, 1)
                else:
                    beam_memory_mask = None
                
                # Create empty cache if needed
                cache = [beam["cache"] for beam in beams]
                if cache[0] is None:
                    cache = None
                
                # Forward through model
                logits, new_cache = self.forward_one_step(
                    ys, tgt_mask, beam_memory, beam_memory_mask, cache=cache
                )
                
                # Get log probabilities for last step
                logp = F.log_softmax(logits[:, -1], dim=-1)
                
                # Update all beams
                new_beams = []
                for i, beam in enumerate(beams):
                    yseq = beam["yseq"]
                    score = beam["score"]
                    
                    # Get logp for current beam
                    beam_logp = logp[i]
                    
                    # Get top k next tokens and their scores
                    topk_logp, topk_indices = beam_logp.topk(beam_size)
                    
                    # Create new beams
                    for k in range(beam_size):
                        new_token = topk_indices[k].item()
                        new_score = score + topk_logp[k].item()
                        
                        # Create new sequence
                        new_yseq = yseq.copy()
                        new_yseq.append(new_token)
                        
                        # Create new cache for this beam
                        if new_cache is not None:
                            beam_new_cache = [c[i:i+1] for c in new_cache]
                        else:
                            beam_new_cache = None
                        
                        # Add new beam
                        new_beams.append({
                            "score": new_score,
                            "yseq": new_yseq,
                            "cache": beam_new_cache
                        })
                
                # Keep top beam_size beams
                beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)[:beam_size]
                
                # Check if all beams end with EOS (assuming 2 is EOS)
                # If so, keep only complete hypotheses
                if all(beam["yseq"][-1] == 2 for beam in beams):
                    break
            
            # Add results for this batch item
            all_results.append(beams)
        
        return all_results 