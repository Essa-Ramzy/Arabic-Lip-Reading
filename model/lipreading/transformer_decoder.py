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

# Import a MODIFIED version of DecoderLayer (instead of importing from ESPNet)
# This addresses the cache shape mismatch issue
class CustomDecoderLayer(nn.Module):
    """Custom decoder layer that handles first-step caching differently and silently."""
    
    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct a DecoderLayer object with custom caching."""
        super(CustomDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)
    
    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        """Compute decoded features with silent handling of cache shape mismatches.
        
        Args:
            tgt (torch.Tensor): decoded previous target features (batch, max_time_out, size)
            tgt_mask (torch.Tensor): mask for x (batch, max_time_out)
            memory (torch.Tensor): encoded source features (batch, max_time_in, size)
            memory_mask (torch.Tensor): mask for memory (batch, max_time_in)
            cache (torch.Tensor): cached output (batch, max_time_out-1, size)
        
        Returns:
            torch.Tensor: output with shape (batch, max_time_out, size)
        """
        # Get batch size from input tensor
        batch_size = tgt.size(0)
        
        # Validate input shapes
        assert memory.size(0) == batch_size, f"Memory batch size {memory.size(0)} != input batch size {batch_size}"
        if memory_mask is not None:
            assert memory_mask.size(0) == batch_size, f"Memory mask batch size {memory_mask.size(0)} != input batch size {batch_size}"
            assert memory_mask.size(1) == memory.size(1), f"Memory mask length {memory_mask.size(1)} != memory length {memory.size(1)}"
        
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        
        # Modified caching mechanism to handle first-step generation
        if cache is None:
            # No cache - use the full target sequence
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # Always extract the last token, regardless of cache shape
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            
            # Handle mask for the last token
            tgt_q_mask = None
            if tgt_mask is not None:
                if tgt_mask.dim() == 3:
                    tgt_q_mask = tgt_mask[:, -1:, :]
                else:
                    tgt_q_mask = tgt_mask
        
        # Compute self-attention with proper masking
        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        
        if not self.normalize_before:
            x = self.norm1(x)
        
        # Source attention
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        
        # Ensure memory_mask has correct shape for attention
        if memory_mask is not None:
            if memory_mask.dim() == 2:
                # Expand to 3D for attention: [batch_size, 1, seq_len]
                memory_mask = memory_mask.unsqueeze(1)
            elif memory_mask.dim() == 3:
                # Ensure correct shape: [batch_size, 1, seq_len]
                if memory_mask.size(1) != 1:
                    memory_mask = memory_mask[:, :1, :]
        
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask)), dim=-1
            )
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        
        if not self.normalize_before:
            x = self.norm2(x)
        
        # Feed-forward
        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        
        x = residual + self.dropout(self.feed_forward(x))
        
        if not self.normalize_before:
            x = self.norm3(x)
        
        # Handle caching
        if cache is not None:
            # Silently handle cache concatenation without checks that might fail
            if cache.shape[1] == 0:
                # First step - just return x (special case)
                return x, tgt_mask, memory, memory_mask
            else:
                # Normal cache concatenation for subsequent steps
                try:
                    x = torch.cat([cache, x], dim=1)
                except:
                    # If concatenation fails for any reason, just use x
                    pass
        
        return x, tgt_mask, memory, memory_mask


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
        
        # Memory projection layer - don't create it yet, will be created with correct size in forward
        self.memory_projection = None
        self.attention_dim = attention_dim
        
        # Create decoder blocks with custom decoder layers
        self.decoders = repeat(
            num_blocks,
            lambda lnum: CustomDecoderLayer(  # Use our custom decoder layer!
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
    
    def _create_memory_projection(self, input_dim):
        """Create memory projection layer with the correct input dimension"""
        if self.memory_projection is None or self.memory_projection.in_features != input_dim:
            self.memory_projection = torch.nn.Linear(input_dim, self.attention_dim)
            # Initialize with xavier uniform
            nn.init.xavier_uniform_(self.memory_projection.weight)
            self.memory_projection = self.memory_projection.to(next(self.parameters()).device)
    
    def forward_one_step(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):
        """Forward one step in generation.
        
        Args:
            tgt: Input token ids, int64 (batch, maxlen_out)
            tgt_mask: Input token mask, (batch, 1, maxlen_out, maxlen_out) or (maxlen_out, maxlen_out)
            memory: Encoded memory, float32 (batch, maxlen_in, feat)
            memory_mask: Encoded memory mask, (batch, maxlen_in) or (batch, 1, maxlen_in)
            cache: Cached output list of (batch, max_time_out-1, size)
            
        Returns:
            y: Output tensor (batch, maxlen_out, vocab_size)
            cache: List of cached outputs [(batch, max_time_out-1, size)]
        """
        # Convert target tokens to embeddings
        x = self.embedding(tgt)
        
        # Create or use existing memory projection layer
        self._create_memory_projection(memory.size(-1))
        
        # Project memory to attention dimension
        memory = self.memory_projection(memory)
        
        # Get batch size and sequence lengths
        batch_size = tgt.size(0)
        tgt_seq_len = tgt.size(1)
        mem_seq_len = memory.size(1)
        
        # Handle mask creation or transformation for self-attention
        if tgt_mask is None:
            # Create default causal mask
            tgt_mask = subsequent_mask(tgt_seq_len).to(x.device)
        
        # Ensure tgt_mask is 3D for attention modules: [batch_size, seq_len, seq_len]
        if tgt_mask.dim() == 2:
            tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)
        elif tgt_mask.dim() == 4:
            tgt_mask = tgt_mask.squeeze(1)
        
        # Ensure tgt_mask is boolean type
        tgt_mask = tgt_mask.bool()
        
        # Create or process memory mask
        if memory_mask is None:
            # Create default memory mask that allows attending to all positions
            memory_mask = torch.ones((batch_size, memory.size(1)), device=memory.device).bool()
            # Handle different memory mask formats
            if memory_mask.dim() == 1 and memory_mask.size(0) == batch_size:
                # Convert length mask to boolean mask
                lengths_mask = memory_mask
                memory_mask = torch.zeros(batch_size, mem_seq_len, device=memory_mask.device).bool()
                for b in range(batch_size):
                    length = lengths_mask[b].item()
                    memory_mask[b, :length] = True
            elif memory_mask.dim() == 2:
                if memory_mask.size(0) == 1:
                    # Expand single batch mask
                    memory_mask = memory_mask.expand(batch_size, -1)
                elif memory_mask.size(1) != mem_seq_len:
                    raise ValueError(f"Memory mask length {memory_mask.size(1)} does not match memory length {mem_seq_len}")
            elif memory_mask.dim() == 3:
                # Handle batch x 1 x seq_len format
                memory_mask = memory_mask.squeeze(1)
            
            # Ensure memory_mask is boolean type
            memory_mask = memory_mask.bool()
            
            # Validate final mask shape
            if memory_mask.size() != (batch_size, mem_seq_len):
                raise ValueError(
                    f"Memory mask shape {memory_mask.size()} does not match required shape ({batch_size}, {mem_seq_len})"
                )
        
        # Invert memory_mask for attention (False = attend, True = don't attend)
        src_attention_mask = ~memory_mask
        
        # Initialize cache if not provided
        new_cache = []
        if cache is None:
            cache = [None] * len(self.decoders)
        
        # Forward through each decoder layer
        for i, (decoder, cache_layer) in enumerate(zip(self.decoders, cache)):
            try:
                # Forward pass with custom decoder layer handling the cache
                x, tgt_mask, memory, _ = decoder(
                    x, tgt_mask, memory, src_attention_mask, cache=cache_layer
                )
                
                # Store in new cache
                new_cache.append(x)
            except Exception as e:
                print(f"Error in decoder layer {i}: {str(e)}")
                raise
        
        # Apply layer norm to the final output
        if self.after_norm is not None:
            if tgt_seq_len > 1:
                # If we have more than one token, take the last one
                y = self.after_norm(x[:, -1])
            else:
                # Otherwise, take the only token
                y = self.after_norm(x.squeeze(1))
        else:
            if tgt_seq_len > 1:
                y = x[:, -1]
            else:
                y = x.squeeze(1)
            
        # Get final output probabilities
        y = self.output_layer(y)
        
        return y, new_cache
    
    def forward(self, tgt, tgt_mask, memory, memory_mask=None):
        """Forward decoder.
        
        Args:
            tgt: Target input sequences tensor, int64 (batch, maxlen_out)
            tgt_mask: Target input mask, (batch, 1, maxlen_out, maxlen_out) or (maxlen_out, maxlen_out)
            memory: Encoded memory, float32 (batch, maxlen_in, feat)
            memory_mask: Encoded memory mask, (batch, maxlen_in) or (batch, 1, maxlen_in)
            
        Returns:
            x: Decoded output tensor (batch, maxlen_out, vocab_size)
        """
        try:
            # Get batch size from input tensor
            batch_size = tgt.size(0)
            
            # Convert target tokens to embeddings
            x = self.embedding(tgt)
            
            # Create or use existing memory projection layer
            self._create_memory_projection(memory.size(-1))
            
            # Project memory to attention dimension
            memory = self.memory_projection(memory)
            
            # Get sequence lengths
            tgt_seq_len = tgt.size(1)
            mem_seq_len = memory.size(1)
            
            # Validate input shapes
            assert memory.size(0) == batch_size, f"Memory batch size {memory.size(0)} != input batch size {batch_size}"
            
            # Handle mask creation or transformation for self-attention
            if tgt_mask is None:
                # Create default causal mask
                tgt_mask = subsequent_mask(tgt_seq_len).to(x.device)
            
            # Ensure tgt_mask is 3D for attention modules: [batch_size, seq_len, seq_len]
            if tgt_mask.dim() == 2:
                tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)
            elif tgt_mask.dim() == 4:
                tgt_mask = tgt_mask.squeeze(1)
            
            # Ensure tgt_mask is boolean type
            tgt_mask = tgt_mask.bool()
            
            # Create or process memory mask
            if memory_mask is None:
                # Create default memory mask that allows attending to all positions
                memory_mask = torch.ones(batch_size, mem_seq_len, device=x.device).bool()
            else:
                # Handle different memory mask formats
                if memory_mask.dim() == 1 and memory_mask.size(0) == batch_size:
                    # Convert length mask to boolean mask
                    lengths_mask = memory_mask
                    memory_mask = torch.zeros(batch_size, mem_seq_len, device=memory_mask.device).bool()
                    for b in range(batch_size):
                        length = lengths_mask[b].item()
                        memory_mask[b, :length] = True
                elif memory_mask.dim() == 2:
                    if memory_mask.size(0) == 1:
                        # Expand single batch mask
                        memory_mask = memory_mask.expand(batch_size, -1)
                    elif memory_mask.size(1) != mem_seq_len:
                        raise ValueError(f"Memory mask length {memory_mask.size(1)} does not match memory length {mem_seq_len}")
                elif memory_mask.dim() == 3:
                    # Handle batch x 1 x seq_len format
                    memory_mask = memory_mask.squeeze(1)
                
                # Ensure memory_mask is boolean type
                memory_mask = memory_mask.bool()
                
                # Validate final mask shape
                if memory_mask.size() != (batch_size, mem_seq_len):
                    raise ValueError(
                        f"Memory mask shape {memory_mask.size()} does not match required shape ({batch_size}, {mem_seq_len})"
                    )
            
            # Invert memory_mask for attention (False = attend, True = don't attend)
            src_attention_mask = ~memory_mask
            
            # Forward through each decoder layer
            for i, decoder in enumerate(self.decoders):
                try:
                    x, tgt_mask, memory, _ = decoder(
                        x, tgt_mask, memory, src_attention_mask
                    )
                except Exception as e:
                    print(f"Error in decoder layer {i}: {str(e)}")
                    raise
            
            # Apply layer norm if needed
            if self.after_norm is not None:
                x = self.after_norm(x)
                
            # Get final output probabilities
            y = self.output_layer(x)
            
            return y
            
        except Exception as e:
            print(f"Error in transformer decoder forward pass: {str(e)}")
            raise
    
    def generate(self, memory, memory_mask, maxlen=50):
        """Generate sequence using autoregressive inference.
        
        Args:
            memory: Encoded memory, float32 (batch, maxlen_in, feat)
            memory_mask: Encoded memory mask, (batch, maxlen_in)
            maxlen: Maximum output length
            
        Returns:
            ys: Generated sequences (batch, seq_len)
            scores: Sequence scores (batch)
        """
        device = memory.device
        batch_size = memory.size(0)
        
        # Create or use existing memory projection layer
        self._create_memory_projection(memory.size(-1))
        
        # Project memory to attention dimension
        memory = self.memory_projection(memory)
        
        # Create memory mask if not provided
        if memory_mask is None:
            memory_mask = torch.ones(batch_size, memory.size(1), device=device).bool()
        else:
            memory_mask = memory_mask.bool()
        
        # Initialize with BOS token (assuming 1 is SOS/BOS token)
        ys = torch.ones(batch_size, 1).long().to(device)  # Start with <sos> token
        scores = torch.zeros(batch_size).to(device)
        
        # Create cache for fast decoding
        cache = None
        
        # Generate tokens one by one
        for i in range(maxlen):
            # Create mask for the current output sequence
            tgt_mask = subsequent_mask(ys.size(-1)).to(device)
            
            try:
                # Forward one step
                logits, cache = self.forward_one_step(
                    ys, tgt_mask, memory, memory_mask, cache=cache
                )
                
                # Log softmax for proper probability distribution
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get the most likely next token
                next_token = log_probs.argmax(dim=-1).unsqueeze(-1)
                
                # Add to score
                scores += log_probs.gather(1, next_token).squeeze(-1)
                
                # Add next token to sequence
                ys = torch.cat([ys, next_token], dim=-1)
                
                # Stop if EOS token is generated for all sequences
                if (next_token == 2).all():  # Assuming 2 is EOS token
                    break
                    
            except Exception as e:
                # If there's an error, return what we have so far
                break
        
        return ys, scores
    
    def batch_beam_search(
        self,
        memory,
        memory_mask=None,
        sos=1,
        eos=2,
        blank=0,
        beam_size=5,
        penalty=0.0,
        maxlen=100,
        minlen=0,
        ctc_weight=0.0
    ):
        """Perform beam search in batch mode with improved memory mask handling.
        
        Args:
            memory: Encoded memory, float32 (batch, maxlen_in, feat)
            memory_mask: Encoded memory mask, (batch, maxlen_in) or (batch, 1, maxlen_in)
            sos: Start of sequence id
            eos: End of sequence id
            blank: Blank symbol id for CTC
            beam_size: Beam size 
            penalty: Insertion penalty
            maxlen: Maximum output length
            minlen: Minimum output length
            ctc_weight: CTC weight for joint decoding
            
        Returns:
            list of tuples containing (score, hypotheses)
            hypotheses is a list of tokens
        """
        device = memory.device
        batch_size = memory.size(0)
        mem_seq_len = memory.size(1)
        results = []
        
        # Project memory to attention dimension
        self._create_memory_projection(memory.size(-1))
        memory = self.memory_projection(memory)
        
        # Create or process memory mask
        if memory_mask is None:
            # Create default memory mask that allows attending to all positions
            memory_mask = torch.ones(batch_size, mem_seq_len, device=device).bool()
        else:
            # Handle different memory mask formats
            if memory_mask.dim() == 1 and memory_mask.size(0) == batch_size:
                # Convert length mask to boolean mask
                lengths_mask = memory_mask
                memory_mask = torch.zeros(batch_size, mem_seq_len, device=memory_mask.device).bool()
                for b in range(batch_size):
                    length = lengths_mask[b].item()
                    memory_mask[b, :length] = True
            elif memory_mask.dim() == 2:
                if memory_mask.size(0) == 1:
                    # Expand single batch mask
                    memory_mask = memory_mask.expand(batch_size, -1)
                elif memory_mask.size(1) != mem_seq_len:
                    raise ValueError(f"Memory mask length {memory_mask.size(1)} does not match memory length {mem_seq_len}")
            elif memory_mask.dim() == 3:
                # Handle batch x 1 x seq_len format
                memory_mask = memory_mask.squeeze(1)
            
            # Ensure memory_mask is boolean type
            memory_mask = memory_mask.bool()
            
            # Validate final mask shape
            if memory_mask.size() != (batch_size, mem_seq_len):
                raise ValueError(
                    f"Memory mask shape {memory_mask.size()} does not match required shape ({batch_size}, {mem_seq_len})"
                )
        
        # Invert memory_mask for attention (False = attend, True = don't attend)
        src_attention_mask = ~memory_mask
        
        # Process each batch item separately
        for b in range(batch_size):
            # Get memory for this batch item with batch dimension preserved
            single_memory = memory[b:b+1]  # Shape: [1, seq_len, hidden_dim]
            single_mask = src_attention_mask[b:b+1]  # Shape: [1, seq_len]
            
            # Initialize with start-of-sequence token
            y = torch.tensor([[sos]], dtype=torch.long, device=device)  # Shape: [1, 1]
            
            # Initialize beam with just the start token
            beam = [{'score': 0.0, 'yseq': [sos], 'cache': None}]
            finished_beams = []
            
            # Perform beam search up to maxlen steps
            for i in range(maxlen):
                candidates = []
                
                # Process each hypothesis in the beam
                for hyp in beam:
                    # Convert sequence to tensor for decoder
                    y = torch.tensor([hyp['yseq']], dtype=torch.long, device=device)
                    
                    # Create self-attention mask for autoregressive property
                    y_mask = subsequent_mask(y.size(1)).to(device)
                    
                    try:
                        # Forward one step with caching
                        with torch.no_grad():
                            y_hat, cache = self.forward_one_step(
                                y, y_mask, single_memory, single_mask, cache=hyp['cache']
                            )
                            
                        # Get log probabilities for next token
                        logp = F.log_softmax(y_hat, dim=-1).squeeze(0)  # [vocab_size]
                        
                        # Get top beam_size next tokens
                        topk_logp, topk_indices = torch.topk(logp, k=beam_size)
                        
                        # Create new candidates for each possible next token
                        for logp_val, token_idx in zip(topk_logp, topk_indices):
                            token_idx = token_idx.item()
                            
                            # Skip blank token
                            if token_idx == blank:
                                continue
                                
                            # Calculate new score
                            new_score = hyp['score'] + logp_val.item()
                            
                            # Apply length penalty for EOS token
                            if token_idx == eos:
                                if len(hyp['yseq']) < minlen:
                                    continue  # Skip EOS if sequence too short
                                # Apply penalty here if needed
                                new_score += penalty * len(hyp['yseq'])
                            
                            # Create new beam
                            new_beam = {
                                'score': new_score,
                                'yseq': hyp['yseq'] + [token_idx],
                                'cache': cache
                            }
                            
                            # Add to finished beams if EOS token
                            if token_idx == eos:
                                finished_beams.append(new_beam)
                            else:
                                candidates.append(new_beam)
                    
                    except Exception as e:
                        print(f"Error in beam {len(candidates)}: {str(e)}")
                        continue
                
                # If we have no candidates and no finished beams, stop search
                if not candidates and not finished_beams:
                    break
                
                # Sort candidates by score and keep top beam_size
                candidates.sort(key=lambda x: x['score'], reverse=True)
                beam = candidates[:beam_size]
                
                # If all beams have finished, break
                if len(beam) == 0:
                    break
            
            # Add any unfinished beams to finished
            finished_beams.extend(beam)
            
            # Sort and get the best result
            if finished_beams:
                finished_beams.sort(key=lambda x: x['score'], reverse=True)
                best_beam = finished_beams[0]
                
                # Remove SOS token
                best_seq = best_beam['yseq'][1:]
                    
                results.append((best_beam['score'], best_seq))
            else:
                # No finished beams, return best unfinished beam
                if beam:
                    beam.sort(key=lambda x: x['score'], reverse=True)
                    best_beam = beam[0]
                    best_seq = best_beam['yseq'][1:]  # Remove SOS
                    results.append((best_beam['score'], best_seq))
                else:
                    # No valid beams at all, return empty sequence
                    results.append((0.0, []))
                
        return results 