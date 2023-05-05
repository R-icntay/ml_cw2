import torch
import torch.nn as nn

# Define a Residual block
class residual_block(nn.Module):
    """
    This class implements a residual block which consists of two convolution layers with group normalization
    """
    def __init__(self, in_channels, out_channels, n_groups = 8):
        super().__init__()
        # First convolution layer
        self.first_conv = nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1, bias=False)
        self.first_norm = nn.GroupNorm(num_groups = n_groups, num_channels = out_channels)
        self.act1 = nn.SiLU() # Swish activation function

        # Second convolution layer
        self.second_conv = nn.Conv3d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, padding = 1, bias = False)
        self.second_norm = nn.GroupNorm(num_groups = n_groups, num_channels = out_channels)
        self.act2 = nn.SiLU() # Swish activation function

        # Add dropout to the residual block
        self.dropout = nn.Dropout3d(p = 0.2)

        # If the number of input channels is not equal to the number of output channels,
        # then use a 1X1 convolution layer to compensate for the difference in dimensions
        # This allows the input to have the same dimensions as the output of the residual block
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, padding = 0, bias = False)
        else:
            # Pass the input as is
            self.shortcut = nn.Identity()

    # Pass the input through the residual block
    def forward(self, x):
        # Store the input
        input = x

        # Pass input through the first convolution layer
        x = self.act1(self.first_norm(self.first_conv(x)))

        # Pass the output of the first convolution layer through the second convolution layer
        x = self.act2(self.second_norm(self.second_conv(x)))

        # Add dropout
        x = self.dropout(x)

        # Add the input to the output of the second convolution layer
        # This is the skip connection
        x = x + self.shortcut(input)
        return x

# Implement the DownSample block that occurs after each residual block
class down_sample(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool3d(kernel_size = 2, stride = 2)

    # Pass the input through the downsample block
    def forward(self, x):
        x = self.max_pool(x)
        return x

# Implement the UpSample block that occurs in the decoder path/expanding path
class up_sample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Convolution transpose layer to upsample the input
        self.up_sample = nn.ConvTranspose3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 2, stride = 2, bias = False)

    # Pass the input through the upsample block
    def forward(self, x):
        x = self.up_sample(x)
        return x

# Implement the crop and concatenate layer
class crop_and_concatenate(nn.Module):
    def forward(self, upsampled, bypass):
        # Crop the upsampled feature map to match the dimensions of the bypass feature map
        if upsampled.shape[2:] != bypass.shape[2:]:
            upsampled = nn.Upsample(size = bypass.shape[2:], mode="trilinear", align_corners=True)(upsampled)

        #upsampled = torchvision.transforms.functional.resize(upsampled, size = bypass.shape[2:], antialias=True)
        x = torch.cat([upsampled, bypass], dim = 1) # Concatenate along the channel dimension
        return x

# Implement an attention block
class attention_block(nn.Module):
    def __init__(self, skip_channels, gate_channels, device, inter_channels = None, n_groups = 8):
        super().__init__()
        self.device = device
        
        if inter_channels is None:
            inter_channels = skip_channels // 2

        # Implement W_g i.e the convolution layer that operates on the gate signal
        # Upsample gate signal to be the same size as the skip connection
        self.W_g = up_sample(in_channels = gate_channels, out_channels = skip_channels)
        self.W_g_norm = nn.GroupNorm(num_groups = n_groups, num_channels = skip_channels)
        self.W_g_act = nn.SiLU() # Swish activation function

        # Implement W_x i.e the convolution layer that operates on the skip connection
        self.W_x = nn.Conv3d(in_channels = skip_channels, out_channels = inter_channels, kernel_size = 1, padding = 0, bias = False)
        self.W_x_norm = nn.GroupNorm(num_groups = n_groups, num_channels = inter_channels)
        self.W_x_act = nn.SiLU() # Swish activation function

        # Implement phi i.e the convolution layer that operates on the output of W_x + W_g
        self.phi = nn.Conv3d(in_channels = inter_channels, out_channels = 1, kernel_size = 1, padding = 0, bias = False)

        # Implement the sigmoid activation function
        self.sigmoid = nn.Sigmoid()
        # Implement the Swish activation function
        self.act = nn.SiLU()

        # Implement final group normalization layer
        self.final_norm = nn.GroupNorm(num_groups = n_groups, num_channels = skip_channels)

    # Pass the input through the attention block
    def forward(self, skip_connection, gate_signal):
        # Upsample the gate signal to match the channels of the skip connection
        gate_signal = self.W_g(gate_signal)
        # Ensure that the sizes of the skip connection and the gate signal match before addition
        
        if gate_signal.shape[2:] != skip_connection.shape[2:]:
            gate_signal = nn.Upsample(size = skip_connection.shape[2:], mode="trilinear", align_corners=True)(gate_signal)

        # Project to the intermediate channels
        gate_signal = self.W_x(gate_signal)

        # Project the skip connection to the intermediate channels
        skip_signal = self.W_x(skip_connection)

        # Add the skip connection and the gate signal
        add_xg = gate_signal + skip_signal

        # Pass the output of the addition through the activation function
        add_xg = self.act(add_xg)

        # Pass the output of attention through a 1x1 convolution layer to obtain the attention map
        attention_map = self.sigmoid(self.phi(add_xg))

        # Multiply the skip connection with the attention map
        # Perform element-wise multiplication
        skip_connection = torch.mul(skip_connection, attention_map)

        skip_connection = nn.Conv3d(in_channels = skip_connection.shape[1], out_channels = skip_connection.shape[1], kernel_size = 1, bias=False).to(self.device)(skip_connection)
        skip_connection = self.act(self.final_norm(skip_connection))

        return skip_connection


## Implement base model
## Implement a MTL 3D resiidual attention U-Net in a more robust manner
class base_MTLResidualAttention3DUnet(nn.Module):
    def __init__(self, in_channels, main_out_channels, device, n_channels = [32, 64, 128, 256, 512], gated_attention = False):
        super().__init__()
        
        # Choose whether to use gated attention or not
        self.gated_attention = gated_attention

        # Define the contracting path: residual blocks followed by downsampling
        self.down_conv = nn.ModuleList()
        in_chans = in_channels
        for out_chans in n_channels[:-1]: # Skip the last element since it is the bottleneck
            self.down_conv.append(residual_block(in_chans, out_chans))
            in_chans = out_chans
        
        self.down_samples = nn.ModuleList(down_sample() for _ in range(len(n_channels) - 1))

        # Define the bottleneck residual block
        self.bottleneck = residual_block(in_channels = n_channels[-2], out_channels = n_channels[-1])

        ## ------ Decoder block for segmenting main prostate zones: central, transition, background ------ ##
        # Define the attention blocks
        self.attention_blocks_main = nn.ModuleList()
        for gate_chans, residual_chans in zip(n_channels[::-1][:-1], n_channels[::-1][1:]):
            self.attention_blocks_main.append(attention_block(skip_channels = residual_chans, gate_channels = gate_chans, device=device))
        
        # Define the expanding path: upsample blocks, followed by crop and concatenate, followed by residual blocks
        self.upsamples_main = nn.ModuleList()
        for in_chans, out_chans in zip(n_channels[::-1][:-1], n_channels[::-1][1:]):
            self.upsamples_main.append(up_sample(in_chans, out_chans))
        
        self.concat_main = nn.ModuleList(crop_and_concatenate() for _ in range(len(n_channels) - 1))

        self.up_conv_main = nn.ModuleList()
        for in_chans, out_chans in zip(n_channels[::-1][:-1], n_channels[::-1][1:]):
            self.up_conv_main.append(residual_block(in_chans, out_chans))

        # Final 1X1 convolution layer to produce the output segmentation map:
        # The primary purpose of 1x1 convolutions is to transform the channel dimension of the feature map,
        # while leaving the spatial dimensions unchanged.
        self.final_conv_main = nn.Conv3d(in_channels = n_channels[0], out_channels = main_out_channels, kernel_size = 1, padding = 0, bias = False) 

    # Pass the input through the residual attention U-Net
    # The input is a 5D tensor of shape (batch_size, channels, depth, height, width)
    def forward(self, x):
        # Store the skip connections
        skip_connections = []

        # Pass the input through the contracting path
        for down_conv, down_sample in zip(self.down_conv, self.down_samples):
            x = down_conv(x)
            skip_connections.append(x)
            x = down_sample(x)

        # Pass the output of the contracting path through the bottleneck
        x = self.bottleneck(x)

        # Define the main variables
        x_main = x

        # --- Pass the output of the encoder through the decoder of the main prostate zones --- #
        # Initialize the attention block counter and the skip connection counter
        attn_block_count = 0
        skip_connections_count = len(skip_connections)

        # Pass the output of the attention blocks through the expanding path
        if self.gated_attention:
            for up_sample, concat, up_conv in zip(self.upsamples_main, self.concat_main, self.up_conv_main):
                gated_attn = self.attention_blocks_main[attn_block_count](skip_connections[skip_connections_count - 1], x_main)
                attn_block_count += 1
                skip_connections_count -= 1
                x_main = up_sample(x_main)
                x_main = concat(x_main, gated_attn)
                x_main = up_conv(x_main)
        else:
            for up_sample, concat, up_conv in zip(self.upsamples_main, self.concat_main, self.up_conv_main):
                x_main = up_sample(x_main)
                x_main = concat(x_main, skip_connections[skip_connections_count - 1])
                x_main = up_conv(x_main)
                skip_connections_count -= 1
        
        # Pass the output of the main decoder through the final convolution layer
        x_main = self.final_conv_main(x_main)  # Output segmentation map for the main prostate zones

        
        # Return the output segmentation maps for the main and auxilliary prostate zones
        return x_main
    
    
######## AUXILIARY TASK ########
class MTLResidualAttention3DUnet(nn.Module):
    def __init__(self, in_channels, main_out_channels, aux_out_channels, device, n_channels = [32, 64, 128, 256, 512], gated_attention = False):
        super().__init__()
        
        # Choose whether to use gated attention or not
        self.gated_attention = gated_attention

        # Define the contracting path: residual blocks followed by downsampling
        self.down_conv = nn.ModuleList()
        in_chans = in_channels
        for out_chans in n_channels[:-1]: # Skip the last element since it is the bottleneck
            self.down_conv.append(residual_block(in_chans, out_chans))
            in_chans = out_chans
        
        self.down_samples = nn.ModuleList(down_sample() for _ in range(len(n_channels) - 1))

        # Define the bottleneck residual block
        self.bottleneck = residual_block(in_channels = n_channels[-2], out_channels = n_channels[-1])

        ## ------ Decoder block for segmenting main prostate zones: central, transition, background ------ ##
        # Define the attention blocks
        self.attention_blocks_main = nn.ModuleList()
        for gate_chans, residual_chans in zip(n_channels[::-1][:-1], n_channels[::-1][1:]):
            self.attention_blocks_main.append(attention_block(skip_channels = residual_chans, gate_channels = gate_chans, device=device))
        
        # Define the expanding path: upsample blocks, followed by crop and concatenate, followed by residual blocks
        self.upsamples_main = nn.ModuleList()
        for in_chans, out_chans in zip(n_channels[::-1][:-1], n_channels[::-1][1:]):
            self.upsamples_main.append(up_sample(in_chans, out_chans))
        
        self.concat_main = nn.ModuleList(crop_and_concatenate() for _ in range(len(n_channels) - 1))

        self.up_conv_main = nn.ModuleList()
        for in_chans, out_chans in zip(n_channels[::-1][:-1], n_channels[::-1][1:]):
            self.up_conv_main.append(residual_block(in_chans, out_chans))

        # Final 1X1 convolution layer to produce the output segmentation map:
        # The primary purpose of 1x1 convolutions is to transform the channel dimension of the feature map,
        # while leaving the spatial dimensions unchanged.
        self.final_conv_main = nn.Conv3d(in_channels = n_channels[0], out_channels = main_out_channels, kernel_size = 1, padding = 0, bias = False) # 


        ## ------ Decoder block for segmenting the auxilliary zones: Bladder, Rectum, Seminal vesicle, Neurovascular bundle ------ ##
        # Define the attention blocks
        self.attention_blocks_aux = nn.ModuleList()
        for gate_chans, residual_chans in zip(n_channels[::-1][:-1], n_channels[::-1][1:]):
            self.attention_blocks_aux.append(attention_block(skip_channels = residual_chans, gate_channels = gate_chans, device=device))

        # Define the expanding path: upsample blocks, followed by crop and concatenate, followed by residual blocks
        self.upsamples_aux = nn.ModuleList()
        for in_chans, out_chans in zip(n_channels[::-1][:-1], n_channels[::-1][1:]):
            self.upsamples_aux.append(up_sample(in_chans, out_chans))
        
        self.concat_aux = nn.ModuleList(crop_and_concatenate() for _ in range(len(n_channels) - 1))

        self.up_conv_aux = nn.ModuleList()
        for in_chans, out_chans in zip(n_channels[::-1][:-1], n_channels[::-1][1:]):
            self.up_conv_aux.append(residual_block(in_chans, out_chans))

        # Final 1X1 convolution layer to produce the output segmentation map:
        # The primary purpose of 1x1 convolutions is to transform the channel dimension of the feature map,
        # while leaving the spatial dimensions unchanged.
        self.final_conv_aux = nn.Conv3d(in_channels = n_channels[0], out_channels = aux_out_channels, kernel_size = 1, padding = 0, bias = False)


    # Pass the input through the residual attention U-Net
    # The input is a 5D tensor of shape (batch_size, channels, depth, height, width)
    def forward(self, x):
        # Store the skip connections
        skip_connections = []

        # Pass the input through the contracting path
        for down_conv, down_sample in zip(self.down_conv, self.down_samples):
            x = down_conv(x)
            skip_connections.append(x)
            x = down_sample(x)

        # Pass the output of the contracting path through the bottleneck
        x = self.bottleneck(x)

        # Define the main and auxilliary variables
        x_main = x
        x_aux = x

        # --- Pass the output of the encoder through the decoder of the main prostate zones --- #
        # Initialize the attention block counter and the skip connection counter
        attn_block_count = 0
        skip_connections_count = len(skip_connections)

        # Pass the output of the attention blocks through the expanding path
        if self.gated_attention:
            for up_sample, concat, up_conv in zip(self.upsamples_main, self.concat_main, self.up_conv_main):
                gated_attn = self.attention_blocks_main[attn_block_count](skip_connections[skip_connections_count - 1], x_main)
                attn_block_count += 1
                skip_connections_count -= 1
                x_main = up_sample(x_main)
                x_main = concat(x_main, gated_attn)
                x_main = up_conv(x_main)
        else:
            for up_sample, concat, up_conv in zip(self.upsamples_main, self.concat_main, self.up_conv_main):
                x_main = up_sample(x_main)
                x_main = concat(x_main, skip_connections[skip_connections_count - 1])
                x_main = up_conv(x_main)
                skip_connections_count -= 1
        
        # Pass the output of the main decoder through the final convolution layer
        x_main = self.final_conv_main(x_main)  # Output segmentation map for the main prostate zones

        # --- Pass the output of the encoder through the decoder of the auxilliary prostate zones --- #
        # Initialize the attention block counter and the skip connection counter
        attn_block_count = 0
        skip_connections_count = len(skip_connections)

        # Pass the output of the attention blocks through the expanding path
        if self.gated_attention:
            for up_sample, concat, up_conv in zip(self.upsamples_aux, self.concat_aux, self.up_conv_aux):
                gated_attn = self.attention_blocks_aux[attn_block_count](skip_connections[skip_connections_count - 1], x_aux)
                attn_block_count += 1
                skip_connections_count -= 1
                x_aux = up_sample(x_aux)
                x_aux = concat(x_aux, gated_attn)
                x_aux = up_conv(x_aux)
        else:
            for up_sample, concat, up_conv in zip(self.upsamples_aux, self.concat_aux, self.up_conv_aux):
                x_aux = up_sample(x_aux)
                x_aux = concat(x_aux, skip_connections[skip_connections_count - 1])
                x_aux = up_conv(x_aux)
                skip_connections_count -= 1
            
        # Pass the output of the auxilliary decoder through the final convolution layer
        x_aux = self.final_conv_aux(x_aux) # Output segmentation map for the auxilliary prostate zones

        # Return the output segmentation maps for the main and auxilliary prostate zones
        return x_main, x_aux