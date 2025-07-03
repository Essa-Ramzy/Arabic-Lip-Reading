import torch
from encoders.encoder_models import VisualTemporalEncoder
from espnet.decoder.transformer_decoder import TransformerDecoder
from espnet.transformer.mask import subsequent_mask
from espnet.transformer.add_sos_eos import add_sos_eos
from espnet.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets_utils import make_non_pad_mask
from espnet.ctc import CTC
from espnet.scorers.ctc import CTCPrefixScorer
from espnet.beam_search import BeamSearch
from torch.nn.utils.rnn import pad_sequence
from espnet.e2e_asr_conformer import E2E as BaseE2E
import logging

class E2EVSR(BaseE2E):
    """
    End-to-end VSR system combining frontend, optional temporal encoder,
    CTC head, and transformer decoder with label-smoothing using beam search decoding.
    """
    def __init__(
        self,
        encoder_type,
        vocab_size,
        token_list,
        sos,
        eos,
        pad,  # This will be your BLANK_ID (0) from the config
        enc_options,
        dec_options,
        ctc_weight=0.3,
        label_smoothing=0.2,
    ):
        # The `pad` argument from your config is the blank_id.
        # The `ignore_id` for ESPnet losses should be -1 for robustness.
        self.pad = pad  # This is the BLANK_ID, which is 0.
        ignore_id = -1  # Define the internal padding/ignore ID.

        # Pass the internal ignore_id to the base class and loss functions.
        super().__init__(odim=vocab_size, modality='video',
                         ctc_weight=ctc_weight, ignore_id=ignore_id)

        # Build one VisualTemporalEncoder model and extract its components
        tcn_opt = enc_options.get('mstcn_options', {}) if encoder_type == 'mstcn' else {}
        dense_opt = enc_options.get('densetcn_options', {}) if encoder_type == 'densetcn' else {}
        conf_opt = enc_options.get('conformer_options', {}) if encoder_type == 'conformer' else {}
        vt_model = VisualTemporalEncoder(
            tcn_options=tcn_opt,
            densetcn_options=dense_opt,
            conformer_options=conf_opt,
            hidden_dim=enc_options['hidden_dim'],
            num_tokens=vocab_size,
            relu_type=enc_options.get('relu_type', 'swish'),
            frontend3d_dropout_rate=enc_options.get('frontend3d_dropout_rate', 0.1),
            resnet_dropout_rate=enc_options.get('resnet_dropout_rate', 0.1),
            resnet_avg_pool_downsample=enc_options.get('resnet_avg_pool_downsample', False),
        )
        self.frontend = vt_model.visual_frontend
        self.proj_encoder = vt_model.adapter
        self.encoder = vt_model.encoder
        self.encoder_type = encoder_type

        # CTC module: its `blank` is 0 by default, `ignore_id` is -1 by default. This is correct.
        self.ctc = CTC(vocab_size, enc_options['hidden_dim'], dropout_rate=0.0, reduce=True)

        # Attention loss: tell it to ignore -1.
        self.att_loss = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=label_smoothing,
        )
        # Transformer decoder
        self.decoder = TransformerDecoder(
            odim=vocab_size,
            attention_dim=dec_options['attention_dim'],
            attention_heads=dec_options['attention_heads'],
            linear_units=dec_options['linear_units'],
            num_blocks=dec_options['num_blocks'],
            dropout_rate=dec_options['dropout_rate'],
            positional_dropout_rate=dec_options['positional_dropout_rate'],
            self_attention_dropout_rate=dec_options.get('self_attention_dropout_rate', 0.0),
            src_attention_dropout_rate=dec_options.get('src_attention_dropout_rate', 0.0),
            normalize_before=dec_options.get('normalize_before', True),
        )
        # Tokens & weights
        self.sos = sos
        self.eos = eos
        # self.pad is already correctly set to 0 (the blank_id)
        self.ctc_weight = ctc_weight
        self.token_list = token_list

    def forward(self, x, x_lengths, ys=None, ys_lengths=None):
        """
        If ys is provided, run teacher-forcing training and return losses.
        Otherwise run beam search inference and return hypotheses.
        """
        # Inference mode (no teacher forcing): use hybrid beam search
        if ys is None:
            return self.beam_search(x, x_lengths, ctc_weight=0.3)

        # ----- TRAINING MODE -----
        logging.info(f"E2EVSR.forward START -> x shape: {x.shape}, x_lengths: {x_lengths}")

        # 1. Process raw video through frontend and temporal encoder
        memory_mask = make_non_pad_mask(x_lengths).to(x.device).unsqueeze(-2)
        v_feats = self.frontend(x)
        v_feats = self.proj_encoder(v_feats)

        # 2. Prepare padded targets for both CTC and Attention.
        # Reconstruct the list of tensors from the flattened `ys`
        batch = []
        start = 0
        for L in ys_lengths.tolist():
            batch.append(ys[start : start + L])
            start += L
        
        # Pad the batch with the correct internal ignore_id (-1)
        ys_pad = pad_sequence(batch, batch_first=True, padding_value=self.ignore_id)

        # 3. Unified temporal encoder forward and CTC Loss calculation
        # All encoder types now return only hidden features
        if self.encoder_type == 'conformer':
            hidden_feats, _ = self.encoder(v_feats, memory_mask)
        else:
            batch_size = v_feats.size(0)
            hidden_feats = self.encoder(v_feats, x_lengths, batch_size)
        
        # Use ESPnet CTC module consistently for all encoder types
        loss_ctc, _ = self.ctc(hidden_feats, x_lengths, ys_pad)
        
        # 4. Attention loss calculation
        # add_sos_eos correctly handles the ignore_id for padding.
        ys_in, ys_out = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        # Use a 2D mask for the decoder, which is cleaner.
        tgt_mask = subsequent_mask(ys_in.size(1), device=ys_in.device)
        tgt_mask = tgt_mask.unsqueeze(0).expand(ys_in.size(0), -1, -1)
        memory_mask_dec = make_non_pad_mask(x_lengths).to(x.device).unsqueeze(1)
        dec_out = self.decoder(ys_in, tgt_mask, hidden_feats, memory_mask_dec)
        if isinstance(dec_out, tuple):
            dec_out = dec_out[0]
        
        # self.att_loss will correctly ignore the -1 padding in ys_out
        loss_att = self.att_loss(dec_out, ys_out)

        # 5. Combine losses
        loss = self.ctc_weight * loss_ctc + (1.0 - self.ctc_weight) * loss_att
        return {'loss': loss, 'ctc_loss': loss_ctc, 'att_loss': loss_att}

    def beam_search(self, x, x_lengths, beam_size=10, ctc_weight=0.0):
        """
        Unified beam search decoding method.

        This function can perform:
        1. Pure CTC Beam Search (ctc_weight=1.0)
        2. Pure Transformer/Attention Beam Search (ctc_weight=0.0)
        3. Hybrid CTC/Attention Beam Search (0.0 < ctc_weight < 1.0)

        Args:
            x (torch.Tensor): Input video features, shape (B, C, T, H, W).
            x_lengths (torch.Tensor): Lengths of input sequences (B,).
            beam_size (int): The beam width.
            ctc_weight (float): Weight for the CTC scorer.

        Returns:
            List[List[int]]: The cleaned, decoded token sequences for the batch.
        """
        # 0. Sanity Checks
        assert x.dim() == 5, f"Expected x to be 5D (B, C, T, H, W), got {x.dim()}D"
        assert 0.0 <= ctc_weight <= 1.0, "ctc_weight must be in [0.0, 1.0]"

        self.eval()
        device = x.device
    
        # 1. Get Encoder Output
        encoder_mask = make_non_pad_mask(x_lengths).to(device).unsqueeze(-2)
        v_feats = self.frontend(x)
        v_feats = self.proj_encoder(v_feats)

        if self.encoder_type == 'conformer':
            hidden_feats, enc_output_mask = self.encoder(v_feats, encoder_mask)
            hlens = enc_output_mask.squeeze(1).sum(dim=1).long()
        else:
            batch_size = v_feats.size(0)
            hidden_feats = self.encoder(v_feats, x_lengths, batch_size)
            hlens = x_lengths.long()

        hidden_feats = hidden_feats.float()

        # 2. Dynamically Setup Scorers and Weights based on arguments
        scorers = {}
        weights = {}
        
        if ctc_weight > 0:
            scorers["ctc"] = CTCPrefixScorer(self.ctc, self.eos)
            weights["ctc"] = ctc_weight

        if ctc_weight < 1:
            scorers["decoder"] = self.decoder
            weights["decoder"] = 1.0 - ctc_weight
            
        # 3. Determine Pre-beaming strategy
        # If the decoder is used, it's a good candidate for pre-beaming.
        pre_beam_score_key = "decoder" if ctc_weight < 1 else None

        # 4. Instantiate BeamSearch
        beam_search = BeamSearch(
            beam_size=beam_size,
            vocab_size=self.odim,
            weights=weights,
            scorers=scorers,
            token_list=self.token_list,
            sos=self.sos,
            eos=self.eos,
            pre_beam_score_key=pre_beam_score_key,
        ).to(device)
    
        # 5. Run the search for each utterance in the batch
        with torch.no_grad():
            results = []
            for b in range(hidden_feats.size(0)):
                enc_out = hidden_feats[b, : hlens[b], :]
                hyps = beam_search(x=enc_out)
                best_hyp = hyps[0] if hyps else None

                if best_hyp:
                    token_tensor = best_hyp.yseq[1:]
                    eos_positions = (token_tensor == self.eos).nonzero()
                    if eos_positions.size(0) > 0:
                        first_eos_pos = eos_positions[0, 0]
                        cleaned_tensor = token_tensor[:first_eos_pos]
                    else:
                        cleaned_tensor = token_tensor
                    final_tokens = cleaned_tensor.cpu().tolist()
                    results.append(final_tokens)
                else:
                    results.append([])
        
        return results
