import torch.nn as nn
import math
import numpy as np
from lipreading.models.resnet import ResNet, BasicBlock
from lipreading.models.shufflenetv2 import ShuffleNetV2
from lipreading.models.tcn import MultibranchTemporalConvNet
from lipreading.models.densetcn import DenseTemporalConvNet
from espnet.encoder.conformer_encoder import ConformerEncoder
from espnet.nets_utils import make_non_pad_mask


# Function to preserve sequence information for Token-level Seq2Seq recognition
def _sequence_batch(x, lengths, B):
    # Just return the sequence data properly shaped for CTC
    # Each item in batch will have sequence length based on its actual length
    return x  # Keep the sequence information intact - shape (B, T, C)


class DenseTCN(nn.Module):
    def __init__(self, block_config, growth_rate_set, input_size, reduced_size, num_classes,
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
        self.encoder_output = nn.Linear(num_features, num_classes)
        
        # Use sequence_batch instead of average_batch for CTC
        self.consensus_func = _sequence_batch

    def forward(self, x, lengths, B):
        # x is of shape (B, T, C) - need to transpose to (B, C, T) for TCN
        x = x.transpose(1, 2)  # Now (B, C, T)
        
        # Process through TCN trunk
        out = self.encoder_trunk(x)  # Shape (B, C, T)
        
        # Transpose back to (B, T, C) for linear layer
        out = out.transpose(1, 2)  # Now (B, T, C)
        
        # Apply linear layer to each time step
        logits = self.encoder_output(out)  # Shape (B, T, num_classes)
        
        return logits


class MultiscaleTCN(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout=0.2, relu_type='relu', dwpw=False):
        super(MultiscaleTCN, self).__init__()
        self.kernel_sizes = tcn_options['kernel_size']
        self.num_kernels = len(self.kernel_sizes)

        self.num_channels = num_channels
        self.input_size = input_size
        self.num_classes = num_classes
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
        # Final output layer - takes output from all branches (num_channels[-1])
        # The error was assuming num_channels[-1]*self.num_kernels, but the MultibranchTemporalConvNet 
        # already concatenates the channels from each branch
        self.encoder_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _sequence_batch

    def forward(self, x, lengths, B):
        # x has dimension (B, T, C) from the Lipreading model
        # TCN trunk expects (B, C, T), so transpose
        x = x.transpose(1, 2)  # Now (B, C, T)
        
        # Run through TCN trunk
        tcn_out = self.encoder_trunk(x)  # Output is (B, C, T)
        
        # Transpose back to (B, T, C) for sequence processing
        tcn_out = tcn_out.transpose(1, 2)
        
        # Use the consensus function (which is _sequence_batch for CTC)
        # _sequence_batch just returns the sequence data properly shaped
        seq_out = self.consensus_func(tcn_out, lengths, B)
        
        # Get original logits 
        original_logits = self.encoder_output(seq_out)  # (B, T, num_classes)
        
        # Return the actual model predictions
        return original_logits


class Lipreading(nn.Module):
    def __init__(self,
                 modality='video',
                 hidden_dim=256,
                 backbone_type='resnet',
                 num_classes=500,
                 relu_type='swish',
                 tcn_options={},
                 densetcn_options={},
                 conformer_options={},
                 frontend_options={},
                 extract_feats=False,
                ):
        super(Lipreading, self).__init__()
        self.extract_feats = extract_feats
        self.backbone_type = backbone_type
        self.modality = modality
        self.frontend_nout = 64
        self.backend_out = 512
        
        # -- BACKBONE --
        if backbone_type == 'resnet':
            backbone = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
        elif backbone_type == 'shufflenet':
            assert tcn_options['tcn_type'] == 'tcn', \
                "Only tcn is supported with shufflenet backbone"
            backbone = ShuffleNetV2(input_size=96, width_mult=1.0, modality=modality)
            self.frontend_nout = 24
            self.backend_out = 1024
            hidden_dim = 1024
        else:
            raise ValueError("Unsupported backbone_type: {}".format(backbone_type))
            
        # -- FRONT-END --
        if 'tcn' in frontend_options:
            self.frontend = nn.Sequential(
                nn.Conv3d(1, self.frontend_nout, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False),
                nn.BatchNorm3d(self.frontend_nout),
                nn.ReLU(True),
                nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)))
        else:
            self.frontend = nn.Sequential(
                nn.Conv3d(1, self.frontend_nout, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False),
                nn.BatchNorm3d(self.frontend_nout),
                nn.ReLU(True),
                nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)))
            
        # -- BACK-END --
        self.trunk = backbone
        
        # -- TEMPORAL MODULE --
        if tcn_options:
            # Create and initialize TCN
            tcn_class = tcn_options.pop('tcn_type', 'multiscale')
            if tcn_class == 'multiscale':
                num_channels = tcn_options.get('num_channels', [hidden_dim//4]*4)
                
                if hidden_dim != self.backend_out:
                    self.adapter = nn.Linear(self.backend_out, hidden_dim)
                else:
                    self.adapter = nn.Identity()
                
                self.encoder = MultiscaleTCN(
                    input_size=hidden_dim,
                    num_channels=num_channels,
                    num_classes=num_classes,
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
                num_classes=num_classes,
                kernel_size_set=densetcn_options['kernel_size_set'],
                dilation_size_set=densetcn_options['dilation_size_set'],
                dropout=densetcn_options['dropout'],
                relu_type=relu_type,
                squeeze_excitation=densetcn_options.get('squeeze_excitation', False),
            )
            
            if hidden_dim != self.backend_out:
                self.adapter = nn.Linear(self.backend_out, hidden_dim)
            else:
                self.adapter = nn.Identity()
        elif conformer_options:
            # Conformer option
            if hidden_dim != self.backend_out:
                self.adapter = nn.Linear(self.backend_out, hidden_dim)
            else:
                self.adapter = nn.Identity()
                
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
            # Add output projection for classification (per time step)
            self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        B, C, T, H, W = x.size()
        # Process through frontend
        x = self.frontend(x)
        
        # Get new time dimension after frontend
        Tnew = x.shape[2]
        
        # Reshape and permute for ResNet processing
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        
        # Process through ResNet trunk
        x = self.trunk(x)
        
        # Reshape back to sequence form
        x = x.view(B, Tnew, -1)
        
        # Apply adapter if needed
        x = self.adapter(x)
        
        if self.extract_feats:
            return x
            
        # Process through temporal module
        if isinstance(self.encoder, ConformerEncoder):
            # Create padding mask for Conformer
            padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
            # Forward through Conformer
            x, _ = self.encoder(x, padding_mask)
            # Apply output projection
            x = self.output_layer(x)
        else:
            # Forward through TCN or DenseTCN
            x = self.encoder(x, lengths, B)
            
        return x


    def _initialize_weights_randomly(self):
        use_sqrt = True
        if use_sqrt:
            def f(n):
                return math.sqrt( 2.0/float(n) )
        else:
            def f(n):
                return 2.0/float(n)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod( m.kernel_size ) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()


