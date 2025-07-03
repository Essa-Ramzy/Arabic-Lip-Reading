import torch.nn as nn
from encoders.modules.resnet import ResNet, BasicBlock
from encoders.modules.tcn import MultibranchTemporalConvNet
from encoders.modules.densetcn import DenseTemporalConvNet
from espnet.encoder.conformer_encoder import ConformerEncoder
from espnet.nets_utils import make_non_pad_mask
import logging

# Function to preserve sequence information for Token-level Seq2Seq recognition
def _sequence_batch(x, lengths, B):
    # Just return the sequence data properly shaped for CTC
    # Each item in batch will have sequence length based on its actual length
    return x  # Keep the sequence information intact - shape (B, T, C)


class DenseTCN(nn.Module):
    def __init__(self, block_config, growth_rate_set, input_size, reduced_size, num_tokens,
                  kernel_size_set, dilation_size_set, 
                  dropout, relu_type,
                  squeeze_excitation=False,
        ):
        super(DenseTCN, self).__init__()

        num_features = reduced_size + block_config[-1]*growth_rate_set[-1]
        self.encoder_trunk = DenseTemporalConvNet(block_config, growth_rate_set, input_size, reduced_size,
                                          kernel_size_set, dilation_size_set,
                                          dropout=dropout, relu_type=relu_type,
                                          squeeze_excitation=squeeze_excitation,
                                          )
        # Remove built-in CTC head - let ESPnet handle CTC
        # Project TCN features into hidden_dim for consistency
        self.hidden_proj = nn.Linear(num_features, input_size)
        
        # Use sequence_batch instead of average_batch for CTC
        self.consensus_func = _sequence_batch

    def forward(self, x, lengths, B):
        # x is of shape (B, T, C) - need to transpose to (B, C, T) for TCN
        x = x.transpose(1, 2)  # Now (B, C, T)
        
        # Process through TCN trunk
        out = self.encoder_trunk(x)  # Shape (B, C, T)
        
        # Transpose back to (B, T, C) for linear layer
        out = out.transpose(1, 2)  # Now (B, T, C)
        
        # Only return hidden features - let ESPnet CTC handle the rest
        hidden_feats = self.hidden_proj(out)   # (B, T, hidden_dim)
        return hidden_feats


class MultiscaleTCN(nn.Module):
    def __init__(self, input_size, num_channels, num_tokens, tcn_options, dropout=0.2, relu_type='relu', dwpw=False):
        super(MultiscaleTCN, self).__init__()
        self.kernel_sizes = tcn_options['kernel_size']
        self.num_kernels = len(self.kernel_sizes)

        self.num_channels = num_channels
        self.input_size = input_size
        self.num_tokens = num_tokens
        self.dropout = dropout
        self.relu_type = relu_type
        self.dwpw = dwpw

        self.encoder_trunk = MultibranchTemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            tcn_options=tcn_options,
            dropout=dropout,
            relu_type=relu_type,
            dwpw=dwpw
        )
        # Remove built-in CTC head - let ESPnet handle CTC
        # Project TCN features into hidden_dim for consistency
        self.hidden_proj = nn.Linear(num_channels[-1], input_size)

        self.consensus_func = _sequence_batch

    def forward(self, x, lengths, B):
        # x has dimension (B, T, C) from the VisualTemporalEncoder model
        # TCN trunk expects (B, C, T), so transpose
        x = x.transpose(1, 2)  # Now (B, C, T)
        
        # Run through TCN trunk
        tcn_out = self.encoder_trunk(x)  # Output is (B, C, T)
        
        # Transpose back to (B, T, C) for sequence processing
        tcn_out = tcn_out.transpose(1, 2)
        
        # Use the consensus function (which is _sequence_batch for CTC)
        # _sequence_batch just returns the sequence data properly shaped
        seq_out = self.consensus_func(tcn_out, lengths, B)
        
        # Only return hidden features - let ESPnet CTC handle the rest
        hidden_feats = self.hidden_proj(seq_out)     # (B, T, hidden_dim)
        return hidden_feats


class VisualFrontend(nn.Module):
    def __init__(self, frontend_nout=64, relu_type='prelu', frontend3d_dropout_rate=0.1, resnet_dropout_rate=0.1, resnet_avg_pool_downsample=False):
        super(VisualFrontend, self).__init__()
        
        # Create frontend3D module with correct activation
        if relu_type == 'prelu':
            act_fn = nn.PReLU(num_parameters=frontend_nout)
        elif relu_type == 'swish':
            act_fn = nn.SiLU(inplace=True)
        else:
            act_fn = nn.ReLU(inplace=True)

        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, frontend_nout, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False),
            nn.BatchNorm3d(frontend_nout),
            act_fn,
            nn.Dropout3d(p=frontend3d_dropout_rate),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        )
        
        # Create ResNet trunk with dropout
        self.resnet_trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type, dropout_rate=resnet_dropout_rate, avg_pool_downsample=resnet_avg_pool_downsample)

    def forward(self, x):
        # Debug: VisualFrontend input shape
        logging.info(f"VisualFrontend.forward -> input x shape: {x.shape}")
        B, C, T, H, W = x.size()
        
        # Process through frontend3D - maintains 5D structure
        x = self.frontend3D(x)
        logging.info(f"After frontend3D shape: {x.shape}")  # [B, frontend_nout, T, H//4, W//4]
        
        # Reshape for ResNet processing
        # First, transpose T to get it after batch
        x = x.transpose(1, 2)
        logging.info(f"After transpose shape: {x.shape}")  # [B, T, frontend_nout, H//4, W//4]
        
        # Reshape by combining B and T, keeping other dimensions
        x = x.contiguous().view(B * T, x.size(2), x.size(3), x.size(4))
        logging.info(f"After view shape: {x.shape}")  # [B*T, frontend_nout, H//4, W//4]
        
        # Process through ResNet trunk
        x = self.resnet_trunk(x)
        logging.info(f"After resnet_trunk shape: {x.shape}")  # [B*T, backend_out]
        
        # Reshape back to separate batch and time dimensions
        x = x.view(B, T, -1)
        logging.info(f"VisualFrontend.forward OUTPUT shape: {x.shape}")  # [B, T, backend_out]
        
        return x


class VisualTemporalEncoder(nn.Module):
    def __init__(self,
                 modality='video',
                 hidden_dim=256,
                 backbone_type='resnet',
                 num_tokens=226,
                 relu_type='swish',
                 tcn_options={},
                 densetcn_options={},
                 conformer_options={},
                 extract_feats=False,
                 frontend3d_dropout_rate=0.1,
                 resnet_dropout_rate=0.1,
                 resnet_avg_pool_downsample=False
                ):
        super(VisualTemporalEncoder, self).__init__()
        self.extract_feats = extract_feats
        self.backbone_type = backbone_type
        self.modality = modality
        self.frontend_nout = 64
        self.backend_out = 512
        
        # Create the visual frontend with dropout
        self.visual_frontend = VisualFrontend(
            frontend_nout=self.frontend_nout, 
            relu_type=relu_type,
            frontend3d_dropout_rate=frontend3d_dropout_rate,
            resnet_dropout_rate=resnet_dropout_rate,
            resnet_avg_pool_downsample=resnet_avg_pool_downsample
        )
        
        # -- TEMPORAL MODULE --
        if tcn_options:
            # Create and initialize TCN
            tcn_class = tcn_options.pop('tcn_type', 'multiscale')
            if tcn_class == 'multiscale':
                num_channels = tcn_options.get('num_channels', [hidden_dim//4]*4)
                
                self.adapter = nn.Linear(self.backend_out, hidden_dim)
                
                self.encoder = MultiscaleTCN(
                    input_size=hidden_dim,
                    num_channels=num_channels,
                    num_tokens=num_tokens,
                    tcn_options=tcn_options,
                    dropout=tcn_options.get('dropout', 0.2),
                    relu_type=relu_type,
                    dwpw=tcn_options.get('dwpw', False),
                )
            elif tcn_class == 'tcn':
                raise ValueError("The standard TCN is no longer supported")
            else:
                raise ValueError("Unsupported TCN type: {}".format(tcn_class))
        elif densetcn_options:
            # DenseTCN option
            self.encoder = DenseTCN(
                block_config=densetcn_options['block_config'],
                growth_rate_set=densetcn_options['growth_rate_set'],
                input_size=hidden_dim,
                reduced_size=densetcn_options['reduced_size'],
                num_tokens=num_tokens,
                kernel_size_set=densetcn_options['kernel_size_set'],
                dilation_size_set=densetcn_options['dilation_size_set'],
                dropout=densetcn_options['dropout'],
                relu_type=relu_type,
                squeeze_excitation=densetcn_options.get('squeeze_excitation', False),
            )
            
            self.adapter = nn.Linear(self.backend_out, hidden_dim)
        elif conformer_options:
            # Conformer option
            self.adapter = nn.Linear(self.backend_out, hidden_dim)
                
            self.encoder = ConformerEncoder(
                attention_dim=hidden_dim,
                attention_heads=conformer_options.get('attention_heads', 8),
                linear_units=conformer_options.get('linear_units', 2048),
                num_blocks=conformer_options.get('num_blocks', 6),
                dropout_rate=conformer_options.get('dropout_rate', 0.1),
                positional_dropout_rate=conformer_options.get('positional_dropout_rate', 0.1),
                attention_dropout_rate=conformer_options.get('attention_dropout_rate', 0.0),
                normalize_before=True,
                concat_after=False,
                macaron_style=True,
                use_cnn_module=True,
                cnn_module_kernel=conformer_options.get('cnn_module_kernel', 31),
            )

        # Remove CTC module from here - let E2EVSR handle CTC consistently
        
        # Remove legacy compatibility code
        import torch.nn as _nn
        self.proj_encoder = _nn.Identity()

    def forward(self, x, lengths):
        B, C, T, H, W = x.size()
        
        # Process through frontend - output shape: [B, T, backend_out]
        x = self.visual_frontend(x)
        
        # Apply adapter if needed - maintains shape [B, T, hidden_dim]
        x = self.adapter(x)
        
        if self.extract_feats:
            return x
        
        # Process through temporal module to get hidden features for decoder
        if isinstance(self.encoder, ConformerEncoder):
            padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
            enc_out = self.encoder(x, padding_mask)
            # Conformer returns (hidden_feats, mask)
            x = enc_out[0]
        else:
            # TCN encoders now only return hidden features
            x = self.encoder(x, lengths, B)
        
        return x
