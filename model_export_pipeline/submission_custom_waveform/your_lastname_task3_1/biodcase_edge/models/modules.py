import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn
import torch.nn.functional as F
try:
    from torchsummary import summary
except Exception:  # pragma: no cover - optional developer convenience dependency
    summary = None
try:
    from omegaconf import DictConfig
except Exception:  # pragma: no cover - optional at inference time
    from typing import Any as DictConfig

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, input):
        # gru_input shape: (batch, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(input), dim=1)
        context_vector = torch.sum(attention_weights * input, dim=1)
        return context_vector, attention_weights

class FocusedAttention(nn.Module):
    def __init__(self, hidden_size):
        super(FocusedAttention, self).__init__()
        # Same parameter count as your original attention
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, input):
        # Apply softmax with temperature to sharpen focus
        attention_logits = self.attention(input)
        # Temperature parameter (hardcoded to avoid extra parameters)
        temp = 2.0
        attention_weights = F.softmax(attention_logits * temp, dim=1)

        # Add a slight bias toward the end of words (for detecting "-ward" suffix)
        seq_len = input.size(1)
        position_bias = torch.linspace(0.8, 1.2, seq_len, device=input.device).unsqueeze(0).unsqueeze(2)
        attention_weights = attention_weights * position_bias
        attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True)

        context_vector = torch.sum(attention_weights * input, dim=1)
        return context_vector, attention_weights

class StatefulRNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StatefulRNNLayer, self).__init__()
        self.rnn_cell = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_t=None):
        batch_size, seq_len, _ = x.size()
        if h_t is None:
            h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = torch.tanh(self.rnn_cell(torch.cat([x_t, h_t], dim=1)))
            outputs.append(h_t.unsqueeze(1))

        return torch.cat(outputs, dim=1), h_t

class LightConsonantEnhancer(nn.Module):
    def __init__(self, feature_dim):
        super(LightConsonantEnhancer, self).__init__()
        # Just 2*feature_dim parameters
        self.enhancer = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        # Enhanced features with residual connection (no additional parameters)
        enhanced = self.enhancer(x)
        return x + torch.tanh(enhanced)

class StatefulGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StatefulGRU, self).__init__()
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_t=None):
        batch_size, seq_len, _ = x.size()
        if h_t is None:
            h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        outputs = []
        for t in range(seq_len):
            h_t = self.gru_cell(x[:, t, :], h_t)
            outputs.append(h_t.unsqueeze(1))

        return torch.cat(outputs, dim=1), h_t

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.causal_padding = dilation * (kernel_size - 1)
        self.pad = nn.ConstantPad1d((self.causal_padding, 0), 0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, **kwargs)

    def forward(self, x):
        return self.conv(self.pad(x))

class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]

    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[...,:-self.causal_padding]

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def correct_pad(input_shape, kernel_size):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_shape[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_shape[0] % 2, 1 - input_shape[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return (
        int(correct[1] - adjust[1]),
        int(correct[1]),
        int(correct[0] - adjust[0]),
        int(correct[0]),
    )

def preprocess_input(x, **kwargs):
    return (x / 128.0) - 1

def get_xpansion_factor(t_zero, beta, block_id, num_blocks):
    return (t_zero * beta) * block_id / num_blocks + t_zero * (
        num_blocks - block_id
    ) / num_blocks

class ReLUMax(torch.nn.Module):
    def __init__(self, max_val): # Renamed max to max_val to avoid conflict with builtin
        super(ReLUMax, self).__init__()
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, min=0, max=self.max_val)

class SEBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, h_swish=True):
        super(SEBlock, self).__init__()

        self.se_conv = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        self.se_conv2 = CausalConv1d(
            out_channels, in_channels, kernel_size=1, bias=False, padding=0
        )

        if h_swish:
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = ReLUMax(6)
        self.mult = nnq.FloatFunctional()

    def forward(self, x):
        inp = x
        x = F.adaptive_avg_pool1d(x, 1)
        x = self.se_conv(x)
        x = self.activation(x)
        x = self.se_conv2(x)
        x = torch.sigmoid(x)
        return self.mult.mul(inp, x)

class DepthwiseCausalConv(CausalConv1d):
    def __init__(
        self,
        in_channels,
        depth_multiplier=1,
        kernel_size=3,
        stride=1,
        padding=0, # Padding is handled by CausalConv1d's self.pad
        dilation=1,
        bias=False,
        padding_mode="zeros", # padding_mode is part of CausalConv1d's **kwargs
    ):
        out_channels = in_channels * depth_multiplier
        super().__init__( # Pass args to CausalConv1d
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            # padding=padding, # CausalConv1d handles its own padding
            dilation=dilation,
            groups=in_channels,
            bias=bias
            # padding_mode=padding_mode # Handled by CausalConv1d
        )

class SeparableCausalConv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation_fn=torch.nn.ReLU, # Renamed activation to activation_fn
        kernel_size=3,
        stride=1,
        # padding=0, # Not used directly by CausalConv1d here
        dilation=1,
        bias=True,
        # padding_mode="zeros", # Not used directly by CausalConv1d here
        # depth_multiplier=1, # Not used, depthwise is in_channels -> in_channels
    ):
        super().__init__()

        # Depthwise convolution
        depthwise = DepthwiseCausalConv( # Uses the modified DepthwiseCausalConv
            in_channels=in_channels,
            depth_multiplier=1, # Ensures out_channels = in_channels for depthwise part
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias, # Bias can be true for depthwise
        )

        # Pointwise convolution (1x1)
        pointwise = CausalConv1d( # Use CausalConv1d for the pointwise part
            in_channels=in_channels, # Output of depthwise is in_channels
            out_channels=out_channels,
            kernel_size=1, # 1x1 conv
            stride=1,
            dilation=1, # Dilation usually 1 for pointwise
            bias=bias, # Bias can be true for pointwise
        )

        bn = torch.nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.999)

        self._layers = torch.nn.ModuleList([
            depthwise,
            pointwise, # Swapped order from original text, depthwise then pointwise is standard
            bn,
            activation_fn() # Instantiate the activation function
        ])


    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

class PhiNetCausalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        expansion,
        stride,
        filters,
        has_se,
        block_id=None, # block_id is used to determine if it's an expansion block
        res=True,
        h_swish=True,
        k_size=3,
        dp_rate=0.05,
        divisor=1,
        dilation=1,
    ):
        super(PhiNetCausalConvBlock, self).__init__()
        self.skip_conn = False
        self._layers = torch.nn.ModuleList()

        activation_fn = nn.Hardswish(inplace=True) if h_swish else ReLUMax(6)
        
        expanded_channels = _make_divisible(int(expansion * in_channels), divisor=divisor)

        # Expansion phase (1x1 conv if block_id is present, meaning it's not the first block in a stage)
        if block_id: 
            self._layers.extend([
                CausalConv1d(in_channels, expanded_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm1d(expanded_channels, eps=1e-3, momentum=0.999),
                activation_fn
            ])
        else: # If no block_id, it's the first block, input channels are already what's needed for depthwise
            expanded_channels = in_channels


        # Depthwise convolution
        # Causal padding is handled within DepthwiseCausalConv
        self._layers.extend([
            nn.Dropout1d(dp_rate), # Dropout before depthwise
            DepthwiseCausalConv(
                in_channels=expanded_channels,
                depth_multiplier=1, # DepthwiseCausalConv's out_channels = expanded_channels * 1
                kernel_size=k_size,
                stride=stride,
                bias=False,
                dilation=dilation,
            ),
            nn.BatchNorm1d(expanded_channels, eps=1e-3, momentum=0.999),
            activation_fn # Activation after depthwise normalization
        ])

        if has_se:
            num_reduced_filters = _make_divisible(max(1, int(expanded_channels / 6)), divisor=divisor) # SE on expanded_channels
            self._layers.append(SEBlock(expanded_channels, num_reduced_filters, h_swish=h_swish))

        # Projection phase (1x1 conv)
        self._layers.extend([
            CausalConv1d(expanded_channels, filters, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(filters, eps=1e-3, momentum=0.999)
            # No activation after the final projection, as per MobileNetV2/V3 blocks before residual add
        ])
        
        if res and in_channels == filters and stride == 1:
            self.skip_conn = True
            self.op = nnq.FloatFunctional()

    def forward(self, x):
        inp = x
        for layer in self._layers:
            x = layer(x)
        if self.skip_conn:
            return self.op.add(x, inp)
        return x

class ResidualUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels, # out_channels for ResidualUnit implies filters for PhiNetCausalConvBlock
                 dilation,
                 # pad_mode="reflect" # Not used by CausalConv
                 ):
        super(ResidualUnit, self).__init__()
        # In this ResidualUnit, it seems out_channels is the number of filters *within* the block,
        # and the residual connection implies in_channels == out_channels for the unit itself.
        # The PhiNetCausalConvBlock's 'filters' param should be 'out_channels' of this unit.
        self.layers = nn.Sequential(
            PhiNetCausalConvBlock(in_channels=in_channels, # input to the block
                                  filters=out_channels,    # output of the block
                                  k_size=7,
                                  dilation=dilation,
                                  has_se=True,
                                  expansion=1, # No expansion in these residual units per original text
                                  stride=1,
                                  res=False, # Residual is handled by this class
                                  block_id="res_sub_block1"), # Provide block_id for expansion logic
            PhiNetCausalConvBlock(in_channels=out_channels, # input is output of previous
                                  filters=out_channels,     # output is same as input for residual
                                  k_size=1,                 # 1x1 conv
                                  has_se=True,
                                  expansion=1,
                                  stride=1,
                                  res=False,
                                  block_id="res_sub_block2")
        )

    def forward(self, x):
        return x + self.layers(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels_arg, out_channels_arg, stride): # Renamed to avoid confusion
        super(EncoderBlock, self).__init__()
        # Assuming out_channels_arg is the number of filters for the final PhiNetCausalConvBlock in this encoder stage
        # And in_channels_arg is the input to the first ResidualUnit
        self.layers = nn.Sequential(
            ResidualUnit(
                in_channels=in_channels_arg, # input to first residual
                out_channels=in_channels_arg, # output of residual must match input for skip
                dilation=1
            ),
            ResidualUnit(
                in_channels=in_channels_arg, # input to second residual
                out_channels=in_channels_arg, # output of residual must match input for skip
                dilation=3
            ),
            ResidualUnit(
                in_channels=in_channels_arg, # input to third residual
                out_channels=in_channels_arg, # output of residual must match input for skip
                dilation=9
            ),
            PhiNetCausalConvBlock( # This block does the downsampling and channel increase
                in_channels=in_channels_arg,  # input from residuals
                filters=out_channels_arg,     # target output channels for this encoder stage
                k_size=stride*2,          # kernel related to stride for downsampling
                stride=stride,
                has_se=True,
                expansion=1, # No expansion before downsampling conv
                res=False, # No residual here, it's a transition block
                block_id="enc_transition_block" # Provide block_id
            ),
        )

    def forward(self, x):
        return self.layers(x)

class Encoder(nn.Module): # For raw audio
    def __init__(self, C, D, strides=(4, 5, 16)): # C: initial channels, D: final embedding dim
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            CausalConv1d(in_channels=1, out_channels=C, kernel_size=7), # Initial conv
            EncoderBlock(in_channels_arg=C, out_channels_arg=2*C, stride=strides[0]),
            EncoderBlock(in_channels_arg=2*C, out_channels_arg=4*C, stride=strides[1]),
            EncoderBlock(in_channels_arg=4*C, out_channels_arg=8*C, stride=strides[2]),
            CausalConv1d(in_channels=8*C, out_channels=D, kernel_size=3) # Final embedding
        )
    def forward(self, x):
        if x.ndim == 2: # Expected [batch, time] for raw audio
            x = x.unsqueeze(1) # -> [batch, 1, time] for Conv1d
        return self.layers(x)

class EncoderSpec(nn.Module): # For Spectrograms
    def __init__(self, C, D, n_mel_bins, strides=(4, 5, 16)):
        super(EncoderSpec, self).__init__()
        self.layers = nn.Sequential(
            CausalConv1d(in_channels=n_mel_bins, out_channels=C, kernel_size=7),
            EncoderBlock(in_channels_arg=C, out_channels_arg=2*C, stride=strides[0]),
            EncoderBlock(in_channels_arg=2*C, out_channels_arg=4*C, stride=strides[1]),
            EncoderBlock(in_channels_arg=4*C, out_channels_arg=8*C, stride=strides[2]),
            CausalConv1d(in_channels=8*C, out_channels=D, kernel_size=3)
        )
    def forward(self, x): # Expected [batch, n_mels, time_frames]
        return self.layers(x)


class MatchboxNet(nn.Module):
    def __init__(self, cfg: DictConfig): # Takes Hydra config for matchbox
        super(MatchboxNet, self).__init__()
        
        # Extract parameters from the Hydra config for MatchboxNet structure
        input_channels = cfg.input_channels         # e.g., n_mel_bins
        initial_filters = cfg.initial_filters     # e.g., 64
        block_configs = cfg.blocks                # List of dicts, each defining a block
                                                  # Example block_config:
                                                  #   num_sub_blocks: 2
                                                  #   filters: 32
                                                  #   kernel_size: 3
                                                  #   dilation_pattern: [1, 2] # len must match num_sub_blocks
                                                  #   stride: 1 (only for first sub-block if downsampling)
                                                  #   expansion: 1 
        final_conv_filters1 = cfg.final_conv_filters1 # e.g., 64
        final_conv_kernel1 = cfg.final_conv_kernel1   # e.g., 5
        final_conv_dilation1 = cfg.final_conv_dilation1 # e.g., 2
        
        output_filters = cfg.output_filters           # e.g., 64 (final output channels)
        output_kernel = cfg.output_kernel             # e.g., 1
        
        dropout_rate = cfg.dropout_rate
        use_se = cfg.use_se

        layers = []
        current_channels = input_channels

        # Initial Convolution
        layers.append(PhiNetCausalConvBlock(
            in_channels=current_channels,
            filters=initial_filters,
            k_size=cfg.initial_kernel, # Add initial_kernel to cfg
            stride=cfg.initial_stride, # Add initial_stride to cfg
            dilation=cfg.initial_dilation, # Add initial_dilation to cfg
            has_se=use_se,
            expansion=cfg.initial_expansion, # Add initial_expansion to cfg
            res=False, # No residual for initial block
            block_id="initial_conv"
        ))
        layers.append(nn.BatchNorm1d(initial_filters))
        layers.append(nn.ReLU()) # Using ReLU as common activation
        layers.append(nn.Dropout(dropout_rate))
        current_channels = initial_filters

        # Building Blocks (B1, B2, B3, etc.)
        for i, block_cfg in enumerate(block_configs):
            num_sub_blocks = block_cfg.num_sub_blocks
            block_output_filters = block_cfg.filters # Filters for this block's output
            
            for j in range(num_sub_blocks):
                sub_block_stride = block_cfg.stride if j == 0 else 1 # Stride only in first sub_block of a block
                sub_block_kernel = block_cfg.kernel_size # Can be a list or int
                if isinstance(sub_block_kernel, list): sub_block_kernel = sub_block_kernel[j]
                
                sub_block_dilation = block_cfg.dilation_pattern[j]
                sub_block_expansion = block_cfg.expansion
                
                # Determine if residual connection is possible
                # Residual if input channels == output channels AND stride == 1
                can_res = (current_channels == block_output_filters) and (sub_block_stride == 1)

                layers.append(PhiNetCausalConvBlock(
                    in_channels=current_channels,
                    filters=block_output_filters,
                    k_size=sub_block_kernel,
                    stride=sub_block_stride,
                    dilation=sub_block_dilation,
                    has_se=use_se,
                    expansion=sub_block_expansion,
                    res=can_res, # Enable residual if conditions met
                    block_id=f"block{i}_sub{j}"
                ))
                # PhiNetCausalConvBlock already includes BatchNorm and activation (if not residual end)
                # If it's not a residual block end, PhiNetCausalConvBlock has no final activation.
                # Add activation and dropout if it's not the final layer of the sub-block that would be added to residual
                if not can_res or not PhiNetCausalConvBlock( #Re-check this logic for activation after res
                    in_channels=current_channels,filters=block_output_filters,k_size=1,stride=1, #dummy
                    res=True).skip_conn: # If skip_conn is false, means no activation at end of PhiNet block
                     layers.append(nn.ReLU()) # Add activation
                layers.append(nn.Dropout(dropout_rate))
                current_channels = block_output_filters

        # Final Convolutional Layers
        layers.append(PhiNetCausalConvBlock(
            in_channels=current_channels,
            filters=final_conv_filters1,
            k_size=final_conv_kernel1,
            stride=1,
            dilation=final_conv_dilation1,
            has_se=use_se,
            expansion=cfg.final_expansion1, # Add final_expansion1
            res=False, # Typically no residual for these final layers
            block_id="final_conv1"
        ))
        layers.append(nn.BatchNorm1d(final_conv_filters1))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        current_channels = final_conv_filters1

        layers.append(PhiNetCausalConvBlock(
            in_channels=current_channels,
            filters=output_filters, # This is the final output channel count
            k_size=output_kernel,
            stride=1,
            dilation=1, # Typically 1 for 1x1 conv
            has_se=use_se,
            expansion=cfg.final_expansion2, # Add final_expansion2
            res=False,
            block_id="final_conv2"
        ))
        layers.append(nn.BatchNorm1d(output_filters))
        layers.append(nn.ReLU()) 
        layers.append(nn.Dropout(dropout_rate))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: [batch_size, input_channels, time]
        return self.network(x)


class MatchboxNetSkip(nn.Module):
    """
    MatchboxNet with skip connections for efficient sequence modeling.
    This version is based on the second, more detailed MatchboxNetSkip from the provided text,
    designed to be configured via a Hydra DictConfig.
    """
    def __init__(self, cfg: DictConfig):
        super(MatchboxNetSkip, self).__init__()

        matchbox_cfg = cfg # cfg is expected to be the 'matchbox' sub-config

        input_channels = matchbox_cfg.get('input_features', 64) # Changed 'input_channels' to 'input_features'
        base_filters = matchbox_cfg.get('base_filters', 64)
        num_blocks = matchbox_cfg.get('num_blocks', 3)
        block_filters_list = matchbox_cfg.get('block_filters_list')
        if block_filters_list is None:
            block_filters_list = [base_filters] * num_blocks
        block_filters_list = list(block_filters_list)
        dropout_rate = matchbox_cfg.get('dropout_rate', 0.3)
        use_se = matchbox_cfg.get('use_se', True)
        expansion_factor = matchbox_cfg.get('expansion_factor', 1) # Default expansion
        
        if len(block_filters_list) != num_blocks:
            raise ValueError(f"Length of block_filters_list ({len(block_filters_list)}) must match num_blocks ({num_blocks})")

        sub_blocks_per_block_list = matchbox_cfg.get('sub_blocks_per_block_list')
        if sub_blocks_per_block_list is None:
            sub_blocks_per_block_list = [2] * num_blocks
        sub_blocks_per_block_list = list(sub_blocks_per_block_list)
        if len(sub_blocks_per_block_list) != num_blocks:
            raise ValueError(f"Length of sub_blocks_per_block_list ({len(sub_blocks_per_block_list)}) must match num_blocks ({num_blocks})")

        # Kernel sizes: can be a list of lists (for each sub-block in each block) or a flat list
        kernel_sizes_config = matchbox_cfg.get('kernel_sizes')
        if kernel_sizes_config is None:
            default_kernel = [[3, 3], [3, 3], [3, 3]]
            if num_blocks == len(default_kernel):
                kernel_sizes_config = default_kernel
            else:
                kernel_sizes_config = [default_kernel[i % len(default_kernel)] for i in range(num_blocks)]
        # Dilations: similar structure to kernel_sizes
        dilations_config = matchbox_cfg.get('dilations')
        if dilations_config is None:
            default_dilations = [[1, 2], [4, 2], [1, 1]]
            if num_blocks == len(default_dilations):
                dilations_config = default_dilations
            else:
                dilations_config = [default_dilations[i % len(default_dilations)] for i in range(num_blocks)]

        skip_cfg = matchbox_cfg.get('skip_connections', {})
        self.enable_block_skips = skip_cfg.get('enable_block_skips', True)
        self.enable_sub_block_skips = skip_cfg.get('enable_sub_block_skips', True)
        self.enable_final_skip = skip_cfg.get('enable_final_skip', True)

        initial_conv_cfg = matchbox_cfg.get('initial_conv', {})
        initial_k_size = initial_conv_cfg.get('kernel_size', 5)
        initial_stride = initial_conv_cfg.get('stride', 1) # Often 1 to preserve resolution for GRU
        initial_dilation = initial_conv_cfg.get('dilation', 1)
        
        final_conv1_cfg = matchbox_cfg.get('final_conv1', {})
        final_k_size1 = final_conv1_cfg.get('kernel_size', 5)
        final_filters1 = final_conv1_cfg.get('filters', base_filters) # Default to base_filters
        final_dilation1 = final_conv1_cfg.get('dilation', 2)

        final_conv2_cfg = matchbox_cfg.get('final_conv2', {})
        final_k_size2 = final_conv2_cfg.get('kernel_size', 1)
        final_filters2 = final_conv2_cfg.get('filters', base_filters) # Final output channels
        final_dilation2 = final_conv2_cfg.get('dilation', 1)


        # Initial convolution
        self.initial_conv_module = nn.Sequential(
            PhiNetCausalConvBlock(
                in_channels=input_channels, filters=base_filters, k_size=initial_k_size,
                stride=initial_stride, dilation=initial_dilation, has_se=use_se,
                expansion=expansion_factor, res=False, block_id="initial_conv"
            ),
            nn.BatchNorm1d(base_filters), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        
        current_channels = base_filters
        self.blocks_modulelist = nn.ModuleList()
        self.projections_modulelist = nn.ModuleList() # For block-level skips if channels change

        for i in range(num_blocks):
            block_sub_blocks = nn.ModuleList()
            block_input_channels = current_channels
            num_sub_blocks_current_block = sub_blocks_per_block_list[i]
            output_filters_current_block = block_filters_list[i]

            # Projection for block skip if channels differ from previous block's output to this block's main path
            if self.enable_block_skips and i > 0 and block_input_channels != output_filters_current_block:
                 # This projection is for the *output* of the previous block to match the *output* of the current block
                 # The actual skip connection for block i happens with input from block i-1.
                 # Let's handle projections more carefully within the forward pass if shapes misalign.
                 # For now, assume projections are needed if block_input_channels != output_filters_current_block
                 # This projection will be applied to the *input* of the block if it's different from sub-block output
                self.projections_modulelist.append(
                    nn.Conv1d(block_input_channels, output_filters_current_block, kernel_size=1, stride=1)
                    if block_input_channels != output_filters_current_block else nn.Identity()
                )
            elif i==0 and block_input_channels != output_filters_current_block : # projection from initial_conv to first block
                self.projections_modulelist.append(
                     nn.Conv1d(block_input_channels, output_filters_current_block, kernel_size=1, stride=1)
                )
            else: # No projection needed or first block with matching channels
                self.projections_modulelist.append(nn.Identity())


            for j in range(num_sub_blocks_current_block):
                sub_block_input_channels = current_channels if j == 0 else output_filters_current_block
                
                # Determine kernel size for this sub-block
                k_val = 3 # Default
                if isinstance(kernel_sizes_config[i], list): k_val = kernel_sizes_config[i][j]
                elif isinstance(kernel_sizes_config, list) and len(kernel_sizes_config) > i : k_val = kernel_sizes_config[i] # if flat list per block
                
                # Determine dilation for this sub-block
                d_val = 1 # Default
                if isinstance(dilations_config[i], list): d_val = dilations_config[i][j]
                elif isinstance(dilations_config, list) and len(dilations_config) > i: d_val = dilations_config[i]

                can_res_sub_block = (sub_block_input_channels == output_filters_current_block)

                block_sub_blocks.append(nn.Sequential(
                    PhiNetCausalConvBlock(
                        in_channels=sub_block_input_channels, filters=output_filters_current_block, k_size=k_val,
                        stride=1, dilation=d_val, has_se=use_se, expansion=expansion_factor,
                        res=can_res_sub_block and self.enable_sub_block_skips, # Residual only if enabled AND shapes match
                        block_id=f"block{i}_sub{j}"
                    ),
                    # PhiNetCausalConvBlock with res=True has no final activation. Add it here if not res.
                    # Or if res=True, activation is handled by residual sum then activation.
                    # For simplicity, let PhiNetCausalConvBlock manage its internal activations.
                    # We add BN and Dropout after the block.
                    nn.BatchNorm1d(output_filters_current_block),
                    nn.ReLU(), # Activation after BN
                    nn.Dropout(dropout_rate)
                ))
            self.blocks_modulelist.append(block_sub_blocks)
            current_channels = output_filters_current_block # Output of this block becomes input for next

        # Final Convolutions
        # Projection from last block output to input of final_conv1
        self.final_block_to_conv1_projection = nn.Conv1d(current_channels, final_filters1, kernel_size=1, stride=1) if current_channels != final_filters1 else nn.Identity()

        self.final_conv1_module = nn.Sequential(
            PhiNetCausalConvBlock(
                in_channels=final_filters1, filters=final_filters1, k_size=final_k_size1,
                stride=1, dilation=final_dilation1, has_se=use_se, expansion=expansion_factor,
                res=False, block_id="final_conv1_block" # res=False, skip handled by self.enable_final_skip
            ),
            nn.BatchNorm1d(final_filters1), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        
        # Projection from final_conv1 output to input of final_conv2
        self.conv1_to_conv2_projection = nn.Conv1d(final_filters1, final_filters2, kernel_size=1, stride=1) if final_filters1 != final_filters2 else nn.Identity()

        self.final_conv2_module = nn.Sequential(
            PhiNetCausalConvBlock(
                in_channels=final_filters2, filters=final_filters2, k_size=final_k_size2,
                stride=1, dilation=final_dilation2, has_se=use_se, expansion=expansion_factor,
                res=False, block_id="final_conv2_block"
            ),
            nn.BatchNorm1d(final_filters2), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        self.output_filters = final_filters2 # Store the number of output filters

    def _apply_skip(self, x, skip_connection, projection_layer=None):
        if projection_layer:
            skip_connection = projection_layer(skip_connection)
        
        # Pad if necessary before adding
        if x.shape[2] != skip_connection.shape[2]:
            diff = abs(x.shape[2] - skip_connection.shape[2])
            if x.shape[2] > skip_connection.shape[2]: # x is longer
                skip_connection = F.pad(skip_connection, (0, diff))
            else: # skip_connection is longer
                x = F.pad(x, (0, diff))
        return x + skip_connection

    def forward(self, x):
        # x: [batch, input_channels, time]
        
        # Initial Convolution
        initial_conv_out = self.initial_conv_module(x) # [batch, base_filters, time]
        
        previous_block_output_for_skip = initial_conv_out 
        current_block_path = initial_conv_out # This is the main path being transformed

        for i, block_of_sub_blocks in enumerate(self.blocks_modulelist):
            # Apply projection to the input of the current block's main path if needed
            # This projection ensures channel counts match for the first sub-block
            # self.projections_modulelist[i] projects from previous_block_output_for_skip channels
            # to the expected input channels of block_of_sub_blocks[0] if they differ,
            # or more accurately, projects previous_block_output_for_skip to match output of current block for skip.
            # For now, the projection logic is simpler: project current_block_path to match first sub-block's filters if needed
            projected_skip_source = self.projections_modulelist[i](previous_block_output_for_skip)
            
            # Input to the first sub-block of this block
            current_sub_block_input = projected_skip_source # if i == 0 else current_block_path
            if i > 0 : # For block > 0, current_block_path is the output of previous block
                 current_sub_block_input = self.projections_modulelist[i](current_block_path)


            previous_sub_block_output_for_skip = current_sub_block_input

            for j, sub_block_module in enumerate(block_of_sub_blocks):
                sub_block_out = sub_block_module(previous_sub_block_output_for_skip if j==0 else current_sub_block_path) # Pass correct input 
                
                if self.enable_sub_block_skips and j > 0: # Skip for sub-blocks (except the first)
                    current_sub_block_path = self._apply_skip(sub_block_out, previous_sub_block_output_for_skip)
                else:
                    current_sub_block_path = sub_block_out
                previous_sub_block_output_for_skip = current_sub_block_path
            
            # After all sub-blocks in a block, current_sub_block_path is the output of this block
            if self.enable_block_skips and i >= 0 : # Skip for blocks
                 # previous_block_output_for_skip is the output of the *previous* block (or initial_conv_out for the first block)
                 # projected_skip_source was already projected to match the *output* dimensionality of the current block
                 current_block_path = self._apply_skip(current_sub_block_path, projected_skip_source)
            # else: # No block skip, output is just the processed path
            # current_block_path already holds the output of the sub-blocks processing
            
            previous_block_output_for_skip = current_block_path # Store for the next block's skip

        # Final convolutions
        # Project output of last block to match input of final_conv1
        final_path = self.final_block_to_conv1_projection(previous_block_output_for_skip)
        
        conv1_out = self.final_conv1_module(final_path)
        if self.enable_final_skip:
            final_path = self._apply_skip(conv1_out, self.final_block_to_conv1_projection(previous_block_output_for_skip)) # Skip from before conv1
        else:
            final_path = conv1_out
            
        # Project output of final_conv1_module to match input of final_conv2_module
        final_path_proj_for_conv2 = self.conv1_to_conv2_projection(final_path)
        
        conv2_out = self.final_conv2_module(final_path_proj_for_conv2)
        if self.enable_final_skip:
            # Skip from output of conv1 (which might itself include a skip)
            final_output = self._apply_skip(conv2_out, self.conv1_to_conv2_projection(final_path))
        else:
            final_output = conv2_out
            
        return final_output


class SRNN(nn.Module): # Simplified Stateful RNN - keeping this version if needed
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_rate=0.3):
        super(SRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers # Not directly used in this simple SRNN cell version
        self.dropout_rate = dropout_rate

        self.Wx = nn.Linear(input_size, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.h_t = None

    def reset_hidden_state(self, batch_size, device):
        self.h_t = torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, x): # x shape: [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = x.shape
        if self.h_t is None or self.h_t.size(0) != batch_size:
            self.reset_hidden_state(batch_size, x.device)

        outputs = [] # Store all hidden states if needed, or just return the last
        for t in range(seq_len):
            Wx_out = self.Wx(x[:, t, :])
            Wh_out = self.Wh(self.h_t)
            self.h_t = torch.tanh(Wx_out + Wh_out)
            self.h_t = self.dropout(self.h_t)
            outputs.append(self.h_t.unsqueeze(1)) # Store sequence of hidden states
        
        # Return all hidden states over sequence, and the final hidden state
        return torch.cat(outputs, dim=1), self.h_t 


class HighwayGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0.0, bidirectional=False):
        super(HighwayGRU, self).__init__()
        # Simplified: using nn.GRU and adding a highway-like output
        # For a full Highway GRU, each gate would need a highway component.
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional_factor = 2 if bidirectional else 1

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, 
                          batch_first=batch_first, dropout=dropout, 
                          bidirectional=bidirectional)
        
        # Highway gate for the output of the GRU
        # Input to highway_fc is GRU's output (hidden_size * bidirectional_factor)
        # and GRU's input (input_size) to make the gate data-dependent.
        # Or, simpler: gate depends only on GRU's output.
        self.highway_fc = nn.Linear(hidden_size * self.bidirectional_factor, 
                                    hidden_size * self.bidirectional_factor)

    def forward(self, x, h=None):
        gru_out, h_n = self.gru(x, h)
        # gru_out shape: (batch, seq_len, hidden_size * bidirectional_factor) if batch_first
        #             or (seq_len, batch, hidden_size * bidirectional_factor)
        
        # Highway gate
        highway_gate = torch.sigmoid(self.highway_fc(gru_out))
        
        # Output is a combination of GRU output and a transformed version of it (e.g. identity or another linear)
        # Here, we use highway_gate to blend gru_out with itself (effectively scaling)
        # A more typical highway combines new output with a carry-over of input (x).
        # For simplicity, let's just use the GRU output directly, as "HighwayGRU" might be a misnomer
        # for a standard GRU in this context if no complex gating is added.
        # If the intention was a true highway network on top of GRU states:
        # transformed_input = self.input_transform(x) # if x needs to be same dim as gru_out
        # highway_output = highway_gate * gru_out + (1 - highway_gate) * transformed_input 
        # return highway_output, h_n
        
        # Given the original context, let's assume AttentionLayer takes GRU output directly.
        # The HighwayGRU here seems to be mostly a standard GRU.
        # If a specific highway mechanism is needed, it should be detailed.
        # For now, returning standard GRU outputs.
        return gru_out, h_n


if __name__ == '__main__':
    pass # Placeholder for any test calls
    # Example:
    # cfg_matchbox = DictConfig({
    #     'input_channels': 64, 'base_filters': 32, 
    #     'block_filters_list': [32, 32, 48], 'dropout_rate': 0.1, 'use_se': True,
    #     'expansion_factor': 2, 'num_blocks': 3, 
    #     'sub_blocks_per_block_list': [2, 2, 2],
    #     'kernel_sizes': [[3,3],[3,3],[5,5]], # ks for each sub-block in each block
    #     'dilations': [[1,2],[1,2],[1,2]],    # d for each sub-block in each block
    #     'initial_conv': {'kernel_size': 7, 'stride': 1, 'dilation': 1},
    #     'final_conv1': {'kernel_size': 5, 'filters': 32, 'dilation': 1},
    #     'final_conv2': {'kernel_size': 1, 'filters': 32, 'dilation': 1},
    #     'skip_connections': {'enable_block_skips': True, 'enable_sub_block_skips': True, 'enable_final_skip': True}
    # })
    # model = MatchboxNetSkip(cfg_matchbox)
    # test_input = torch.randn(2, 64, 128) # batch, channels, time
    # out = model(test_input)
    # print("MatchboxNetSkip output shape:", out.shape)

    # attention = AttentionLayer(32* (2 if cfg_matchbox.get('bidirectional_gru', False) else 1) ) # Example hidden dim
    # test_gru_out = torch.randn(2, 50, 32) # batch, seq_len, hidden
    # context, weights = attention(test_gru_out)
    # print("Attention context:", context.shape, "Weights:", weights.shape)
