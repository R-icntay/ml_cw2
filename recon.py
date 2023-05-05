## Implement a MTL 3D resiidual attention U-Net in a more robust manner
class MTLResidualAttention3DUnet(nn.Module):
    def __init__(self, in_channels, main_out_channels, aux_out_channels, n_channels = [32, 64, 128, 256, 512], gated_attention = False):
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
            self.attention_blocks_main.append(attention_block(skip_channels = residual_chans, gate_channels = gate_chans))
        
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
            self.attention_blocks_aux.append(attention_block(skip_channels = residual_chans, gate_channels = gate_chans))

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
    
# Call function
model = MTLResidualAttention3DUnet(in_channels = 1, main_out_channels = 3, aux_out_channels = 1, n_channels = [32, 64, 128, 256, 512], gated_attention = True).to(device)#Main: 2 structures + background, Aux: 3 structures + background