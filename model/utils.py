import torch
import torch.nn.functional as F
import editdistance
import numpy as np



def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

    """
    return ~make_pad_mask(lengths, xs, length_dim)


def pad_packed_collate(batch):
    """Pads data and labels with different lengths in the same batch
    """
    data_list, input_lengths, labels_list, label_lengths = zip(*batch)
    c, max_len, h, w = max(data_list, key=lambda x: x.shape[1]).shape

    data = torch.zeros((len(data_list), c, max_len, h, w))
    
    # Only copy up to the actual sequence length
    for idx in range(len(data)):
        data[idx, :, :input_lengths[idx], :, :] = data_list[idx][:, :input_lengths[idx], :, :]
    
    # Flatten labels for CTC loss
    labels_flat = []
    for label_seq in labels_list:
        labels_flat.extend(label_seq)
    labels_flat = torch.LongTensor(labels_flat)
    
    # Convert lengths to tensor
    input_lengths = torch.LongTensor(input_lengths)
    label_lengths = torch.LongTensor(label_lengths)
    return data, input_lengths, labels_flat, label_lengths

   


def indices_to_text(indices, idx2char):
    """
    Converts a list of indices to text using the reverse vocabulary mapping.
    """
    try:
        return ''.join([idx2char.get(i, '') for i in indices])
    except UnicodeEncodeError:
        # Handle encoding issues in Windows console
        # Return a safe representation that won't cause encoding errors
        safe_text = []
        for i in indices:
            char = idx2char.get(i, '')
            try:
                # Test if character can be encoded
                char.encode('cp1252')
                safe_text.append(char)
            except UnicodeEncodeError:
                # Replace with a placeholder for characters that can't be displayed
                safe_text.append(f"[{i}]")
        return ''.join(safe_text)

def compute_cer(reference_indices, hypothesis_indices):
    """
    Computes Character Error Rate (CER) directly using token indices.
    Takes raw token indices from our vocabulary (class_mapping.txt) rather than Unicode text.
    
    Returns a tuple of (CER, edit_distance)
    """
    # Use the indices directly - each index is one token in our vocabulary
    ref_tokens = reference_indices
    hyp_tokens = hypothesis_indices
    
    try:
        print(f"Debug - Reference tokens ({len(ref_tokens)} tokens): {ref_tokens}")
        print(f"Debug - Hypothesis tokens ({len(hyp_tokens)} tokens): {hyp_tokens}")
    except UnicodeEncodeError:
        # Handle encoding issues in Windows console
        print(f"Debug - Reference tokens ({len(ref_tokens)} tokens): [Token indices omitted due to encoding issues]")
        print(f"Debug - Hypothesis tokens ({len(hyp_tokens)} tokens): [Token indices omitted due to encoding issues]")
    
    # Calculate edit distance using the editdistance library
    edit_distance = editdistance.eval(ref_tokens, hyp_tokens)
    
    # Calculate CER
    cer = edit_distance / max(len(ref_tokens), 1)  # Avoid division by zero
    
    return cer, edit_distance












########################################## NOT USED ##########################################


def compute_cer(reference_indices, hypothesis_indices):
    """
    Computes Character Error Rate (CER) directly using token indices.
    Takes raw token indices from our vocabulary (class_mapping.txt) rather than Unicode text.
    
    Returns a tuple of (CER, reference_len, hypothesis_len, edit_distance)
    """
    # Use the indices directly - each index is one token in our vocabulary
    ref_tokens = reference_indices
    hyp_tokens = hypothesis_indices
    
    print(f"Debug - Reference tokens ({len(ref_tokens)} tokens): {ref_tokens}")
    print(f"Debug - Hypothesis tokens ({len(hyp_tokens)} tokens): {hyp_tokens}")
    
    m, n = len(ref_tokens), len(hyp_tokens)
    
    # Initialize the distance matrix
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Base cases: empty hypothesis or reference
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the distance matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # If tokens match, no operation needed
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Minimum of:
                # 1. Substitution: dp[i-1][j-1] + 1
                # 2. Insertion: dp[i][j-1] + 1
                # 3. Deletion: dp[i-1][j] + 1
                dp[i][j] = min(dp[i - 1][j - 1] + 1,  # substitution
                              dp[i][j - 1] + 1,      # insertion
                              dp[i - 1][j] + 1)      # deletion
    
    edit_distance = dp[m][n]
    cer = edit_distance / max(m, 1)  # Avoid division by zero
    
    return cer, edit_distance




def beam_search_decode(logits, input_lengths=None, beam_size=5, blank_id=0, token_list=None):
    """
    Decode predictions using beam search
    
    Args:
        logits: Logits from the model (B, T, C) or (T, C) for a single sample
        input_lengths: Length of input sequences
        beam_size: Beam size for beam search
        blank_id: ID of blank token
        token_list: List of tokens
        
    Returns:
        decoded: List of decoded sequences
    """
    # Handle single sample case (T, C)
    single_sample = False
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)  # Add batch dimension (1, T, C)
        single_sample = True
        
    if token_list is None:
        token_list = [str(i) for i in range(logits.size(-1))]
        
    batch_size = logits.size(0)
    
    if input_lengths is None:
        input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long, device=logits.device)
    
    decoded = []
    
    # Apply log softmax to get log probabilities
    log_probs = F.log_softmax(logits, dim=2)  # (B, T, C)
    
    # Process each item in the batch
    for b in range(batch_size):
        try:
            # Try to use beam search decoding
            item_log_probs = log_probs[b, :input_lengths[b]]  # (T, C)
            
            # Custom beam search implementation
            T, V = item_log_probs.shape  # Time steps, Vocabulary size
            
            # Initialize beam with just the blank token
            beam = [([], 0.0)]  # (prefix, accumulated_log_prob)
            
            # For each time step
            for t in range(T):
                # Get log probabilities for current time step
                curr_log_probs = item_log_probs[t]  # (V,)
                
                # Collect new candidates
                candidates = []
                
                # Extend each hypothesis in the beam
                for prefix, score in beam:
                    # Option 1: Add new token (except blank)
                    for v in range(V):
                        if v == blank_id:
                            continue
                            
                        # If this token is the same as the last one, it gets merged in CTC
                        if prefix and v == prefix[-1]:
                            new_prefix = prefix.copy()
                            new_score = score + curr_log_probs[v].item()
                        else:
                            new_prefix = prefix + [v]
                            new_score = score + curr_log_probs[v].item()
                            
                        candidates.append((new_prefix, new_score))
                    
                    # Option 2: Add blank (just keep the same prefix)
                    new_score = score + curr_log_probs[blank_id].item()
                    candidates.append((prefix, new_score))
                
                # Sort candidates by score and keep top beam_size
                beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # Get the best hypothesis
            best_hyp, _ = beam[0]
            decoded.append(best_hyp)
            
        except Exception as e:
            # Fallback to greedy decoding if beam search fails
            print(f"Beam search failed with error: {e}, falling back to greedy decoding")
            item_log_probs = log_probs[b, :input_lengths[b]]
            greedy_result = torch.argmax(item_log_probs, dim=1).tolist()
            # Simple CTC decoding: remove consecutive duplicates and blanks
            filtered_tokens = []
            prev_token = -1
            for token in greedy_result:
                if token != blank_id and token != prev_token:
                    filtered_tokens.append(token)
                prev_token = token
            decoded.append(filtered_tokens)
        
            # If input was a single sample, return just the first result
            if single_sample:
                return decoded[0]
            
            return decoded

# %%
# Replace beam search with Transformer decoder inference
def transformer_decode(log_probs, beam_size=8, maxlen=24, blank_index=0):
    """
    Perform transformer-based decoding on log probabilities with additional debugging.
    
    Args:
        log_probs: Log probabilities from the encoder, shape (B, T, C)
        beam_size: Beam width for search
        maxlen: Maximum length of the decoded sequence
        blank_index: Index of the blank token
        
    Returns:
        List of hypotheses, each with 'yseq' and 'score' keys
    """
    batch_size = log_probs.size(0)
    print(f"Starting transformer_decode with batch size: {batch_size}")
    
    # Create memory from encoder features
    memory = log_probs
    
    # Create memory mask (indicates valid encoder positions)
    memory_mask = torch.ones((batch_size, memory.size(1)), device=device)
    
    print(f"Memory shape: {memory.shape}")
    print(f"Memory mask shape: {memory_mask.shape}")
    
    # Debug memory stats
    print(f"Memory stats: mean={memory.mean().item():.4f}, std={memory.std().item():.4f}")
    
    all_results = []
    
    for b in range(batch_size):
        print(f"\nProcessing batch item {b+1}/{batch_size} in beam search")
        # Get this item's memory
        single_memory = memory[b:b+1]  # Keep batch dimension
        single_memory_mask = memory_mask[b:b+1]  # Keep batch dimension
        
        try:
            print("Starting beam search...")
            # Beam search params
            char_list = list(idx2char.keys())  # List of valid character indices
            
            # Beam search initialization
            # Initial state with start token
            y = torch.tensor([1], dtype=torch.long, device=device).reshape(1, 1)  # Start token
            
            # Initialize beam with single hypothesis
            beam = [{'score': 0.0, 'yseq': [1], 'cache': None}]
            
            # Set up length normalization parameters
            length_penalty = 0.6  # Adjust this parameter for better results
            diversity_penalty = 0.1  # Penalty for repeated tokens
            
            for i in range(maxlen):
                print(f"Beam search step {i+1}/{maxlen}")
                if len(beam) == 0:
                    print("Empty beam, breaking")
                    break
                
                # Collect candidates from all beam hypotheses
                new_beam = []
                
                for hyp in beam:
                    # Convert yseq to tensor
                    vy = torch.tensor(hyp['yseq'], dtype=torch.long, device=device).reshape(1, -1)
                    
                    # Create proper causal mask for autoregressive property
                    vy_mask = subsequent_mask(vy.size(1)).to(device)
                    
                    # Forward through transformer decoder
                    try:
                        decoder_out = transformer_decoder(
                            vy,                # Input token sequence
                            vy_mask,           # Self-attention causal mask 
                            single_memory,     # Memory from encoder
                            single_memory_mask  # Memory mask (valid positions)
                        )
                        
                        # Get the last prediction (most recent token)
                        y_logits = decoder_out[:, -1]
                        
                        # Convert to log probs
                        local_scores = F.log_softmax(y_logits, dim=-1)
                        
                        # Add to beam for every possible next token
                        for c in char_list:
                            # Skip blank token
                            if c == blank_index:
                                continue
                                
                            # Apply length normalization to scores
                            normalized_score = (hyp['score'] + local_scores[0, c].item()) / \
                                              ((len(hyp['yseq']) + 1) ** length_penalty)
                            
                            # Apply diversity penalty for repeated tokens
                            if c in hyp['yseq']:
                                normalized_score -= diversity_penalty
                            
                            # Create new hypothesis
                            new_hyp = {
                                'score': normalized_score,
                                'yseq': hyp['yseq'] + [c],
                                'cache': None
                            }
                            
                            new_beam.append(new_hyp)
                    except Exception as e:
                        print(f"Error in decoder forward pass: {str(e)}")
                        continue
                
                # No candidates found
                if len(new_beam) == 0:
                    print("No candidates in new beam, breaking")
                    break
                
                # Sort and keep top beam_size hypotheses
                new_beam.sort(key=lambda x: x['score'], reverse=True)
                beam = new_beam[:beam_size]
                
                # Debug beam status
                print(f"Top beam after step {i+1}:")
                for j, top_hyp in enumerate(beam[:3]):  # Just show top 3
                    print(f"  {j+1}: score={top_hyp['score']:.4f}, seq={top_hyp['yseq']}")
                
                # Check if all beam hypotheses end with EOS
                if all(hyp['yseq'][-1] == 2 for hyp in beam):
                    print("All hypotheses end with EOS, breaking")
                    break
            
            print(f"Beam search complete for batch item {b+1}")
            print(f"Final beam size: {len(beam)}")
            
            # Sort final beam
            beam.sort(key=lambda x: x['score'], reverse=True)
            all_results.append(beam)
            
        except Exception as e:
            print(f"Error during beam search for batch item {b+1}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Add an empty result for this batch
            all_results.append([])
    
    # Return the best hypothesis for the first batch item (simplified)
    if len(all_results) > 0 and len(all_results[0]) > 0:
        return all_results[0]
    else:
        return []
