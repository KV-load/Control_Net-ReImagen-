import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import os
from copy import deepcopy
import torch.utils.checkpoint as checkpoint


# from diffusers import CrossAttention2D

def make_zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

org_unet_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ctrl_unet_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # for reducing memory fragmentation

# Gradient scale
image_scale = 2
text_scale = 3 


    
class ZeroConv2d(nn.Module):  
    def __init__(self, in_ch, out_ch):  
        super().__init__()  
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1,padding=0)  
        nn.init.zeros_(self.conv.weight)  
        nn.init.zeros_(self.conv.bias)  

    def forward(self, x):  
        return self.conv(x)  # Gradients flow through this  


# class Similar_Image_conditioning(nn.Module):
#     def __init__(self, device=device0):
#         super(Similar_Image_conditioning, self).__init__()

#         self.l1 = nn.Conv2d(kernel_size=4, stride=2, in_channels=3, out_channels=16)
#         self.l2 = nn.Conv2d(kernel_size=4, stride=2, in_channels=16, out_channels=32)
#         self.l3 = nn.Conv2d(kernel_size=4, stride=2, in_channels=32, out_channels=64)
#         self.out = nn.Conv2d(kernel_size=4, stride=2, in_channels=64, out_channels=128)

#         self.device = device

#         self.transformer = transforms.Compose([transforms.Resize((1054, 1054)),
#                                   transforms.ToTensor()])

#     def forward(self, x):
#         x = self.transformer(x)
#         x = x.to(self.device)
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         x = F.relu(self.l3(x))
#         out = self.out(x)
#         out = out.unsqueeze(0)
#         return out




class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(embed_dim, embed_dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, query, context):
        """Enhanced cross-attention with residual connections"""
        B, C, H, W = query.shape
        # print(query.shape,"========",context.shape)
        # Reshape for attention
        query_flat = query.view(B, C, H*W).permute(0, 2, 1)
        context_flat = context.view(B, C, H*W).permute(0, 2, 1)
        # scaling_factor = 1 / (query_flat.size(-1) ** 0.5)
        
        # Attention with residual
        attn_out, _ = self.attn(
            query=query_flat,
            key=context_flat,
            value=context_flat
        )
        # attn_out = attn_out * scaling_factor  # Scale attention output
        attn_out = self.norm(attn_out + query_flat)  # Add residual
        
        # Reshape back and project
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        return self.proj(attn_out)


class ReImagen(nn.Module):
    """ControlNet module for SD"""

    def __init__(self, SD_org,Training=True):
        super().__init__()
        self.org_unet = SD_org.to(org_unet_device)

        for param in self.org_unet.parameters():
            param.requires_grad = False  # Freeze original UNet parameters

        
        self.cpy_unet = deepcopy(SD_org).to(ctrl_unet_device)
        self.down_in_channel_resnet = [320, 640, 1280, 1280]
        self.down_out_channel_resnet = [320, 640, 1280, 1280]

        mid_in_channels = [1280]
        mid_out_channels = [1280]

        down_in_channel_mout = [320, 320, 640, 1280]
        down_out_channel_mout = [320, 320, 640, 1280]

        
        self.training = Training  # Set to True for training mode

        self.num_heads = 16

        self.use_checkpointing = False  # Enable gradient checkpointing

        self.condn_drop_prob = 0.1  # Dropout probability for conditioning

        #------------- Null entries for  conditioning--------------#
        self.img_null_entry = nn.Parameter(torch.randn(1, 4, 64, 64), requires_grad=True)  # Null entry for image conditioning
        self.txt_null_entry = nn.Parameter(torch.randn(1,77, 768), requires_grad=True)
        self.clip_null_entry = nn.Parameter(torch.randn(1,77, 768), requires_grad=True)
        #----------------------------------------------------------#


        # Hint block for unet
        # self.ctrl_unet_hint_block = nn.Sequential(
        #     Similar_Image_conditioning(device),
        #     nn.Conv2d(128, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # )
        

        # zero time embedding
        zero_tensor = torch.as_tensor([0], dtype=next(self.cpy_unet.parameters()).dtype, device=ctrl_unet_device)
        self.zero_ctrl_unet_temb = self.cpy_unet.time_embedding(self.cpy_unet.time_proj(zero_tensor))
        self.zero_ctrl_unet_temb = self.zero_ctrl_unet_temb.to(ctrl_unet_device)
            # self.cross_attn = nn.MultiheadAttention()

        self.cross_attn= nn.ModuleList([nn.ModuleList([CrossAttentionBlock(embed_dim=embed_dim, num_heads=self.num_heads)
                                         for embed_dim in self.down_in_channel_resnet]) for _ in range(2)])

        self.cross_attn = self.cross_attn.to(ctrl_unet_device)

        # Converge the text condition and similar image embedding
        self.converge = nn.Sequential(nn.Linear(768*3, 768*2),
                                      nn.SiLU(),
                                      nn.Linear(768*2, 768),
                                      nn.LayerNorm(768),
                                     )
        self.converge = self.converge.to(org_unet_device)



        # Zero Convolution layers for Down blocks
        self.ctrl_unet_down_zero_convs_resnet = nn.ModuleList([
            ZeroConv2d(
                self.down_in_channel_resnet[i],
                self.down_out_channel_resnet[i]
            )
            for i in range(len(self.org_unet.down_blocks))
        ])

        self.ctrl_unet_down_zero_convs_resnet = self.ctrl_unet_down_zero_convs_resnet.to(ctrl_unet_device)
        
        self.ctrl_unet_down_zero_convs_mout = nn.ModuleList([
            ZeroConv2d(
                down_in_channel_mout[i],
                down_out_channel_mout[i],
            )
            for i in range(len(self.org_unet.down_blocks))
        ])

        self.ctrl_unet_down_zero_convs_mout = self.ctrl_unet_down_zero_convs_mout.to(ctrl_unet_device)
        
        # Zero Convolution layer for Mid blocks
        self.ctrl_unet_mid_zero_convs = ZeroConv2d(
            mid_in_channels[0],
            mid_out_channels[0],
            
        )
        self.ctrl_unet_mid_zero_convs = self.ctrl_unet_mid_zero_convs.to(ctrl_unet_device)


        
    def _checkpoint_block(self, block, *args):
        """Wrapper for gradient checkpointing"""
        if self.use_checkpointing and self.training:
            return checkpoint.checkpoint(block, *args, use_reentrant=False)
        return block(*args)
    
    def get_params(self):
        # Add all ControlNet parameters
        params = list(self.cpy_unet.parameters())
        # params = list(self.cpy_unet.down_blocks.parameters())
        # params += list(self.cpy_unet.mid_block.parameters())
        # params += list(self.ctrl_unet_hint_block.parameters())
        params += list(self.ctrl_unet_down_zero_convs_resnet.parameters())
        params += list(self.ctrl_unet_down_zero_convs_mout.parameters())
        params += list(self.ctrl_unet_mid_zero_convs.parameters())
        # params += list(self.converge.parameters())
        params += list(self.cross_attn.parameters())
        params.append(self.img_null_entry)
        params.append(self.txt_null_entry)
        params.append(self.clip_null_entry)
        return params

    def forward(self, x, t, text_condn, hint, similar_img_clip_embed, similar_imgs):
        
        # Collect all downblock residual samples from original UNet
        down_block_res_samples = []
        ctrl_block_t = t.clone().to(ctrl_unet_device)  # Ensure t is on the control UNet device
        ctrl_block_inp = x.clone().to(ctrl_unet_device)
        ctrl_txt_condn = text_condn.clone().to(ctrl_unet_device)
          # Clone t to avoid modifying the original tensor
        
        # org unet entries
        t = t.to(org_unet_device)
        x = x.to(org_unet_device)
        text_condn = text_condn.to(org_unet_device)



        # ------------------ Dropout for conditioning ------------------#

        if self.training and (torch.rand(1)< self.condn_drop_prob): # conditioning on text
                # print("Conditioning dropped")
                text_condn = self.txt_null_entry.to(org_unet_device)
                ctrl_txt_condn = self.txt_null_entry.to(ctrl_unet_device)

        if self.training and (torch.rand(1)< self.condn_drop_prob): # conditioning on similar images
                similar_img_clip_embed = [self.clip_null_entry.to(org_unet_device) for i in range(2)]
                similar_imgs = [self.img_null_entry.to(ctrl_unet_device) for i in range(2)]

        





        # print("process started================",torch.cuda.memory_summary())
        # print(f"Text condition shape: {text_condn.shape}")
    
        with torch.no_grad():
            org_unet_temb = self.org_unet.time_embedding(self.org_unet.time_proj(t))
            org_unet_out = self.org_unet.conv_in(x)
            
            down_block_res_samples.append(org_unet_out)
            # Pass through down blocks
            for down in self.org_unet.down_blocks:
                if hasattr(down, 'attentions'):
                    org_unet_out = down(org_unet_out, org_unet_temb, encoder_hidden_states=text_condn)
                else:
                    org_unet_out = down(org_unet_out, org_unet_temb)
                
                if isinstance(org_unet_out, tuple):
                    org_unet_out, res_samples = org_unet_out
                    down_block_res_samples.extend(res_samples)


        

        # print("down block process done===============",torch.cuda.memory_summary())

        # Collect all downblock residual samples from control UNet
        hint = hint.to(ctrl_unet_device)
        ctrl_unet_temb = self.cpy_unet.time_embedding(self.cpy_unet.time_proj(ctrl_block_t))
        ctrl_unet_hint_out = self.cpy_unet.conv_in(hint)
        
        ctrl_unet_out = self.cpy_unet.conv_in(ctrl_block_inp)
        ctrl_unet_out = ctrl_unet_out + ctrl_unet_hint_out
        
        ctrl_down_block_res_samples = []

        ctrl_down_block_res_samples.append(ctrl_unet_out)

        
        #getting similar images here info
        similar_imgs = [img.to(ctrl_unet_device) for img in similar_imgs]
        similar_img_one = self.cpy_unet.conv_in(similar_imgs[0])
        similar_img_two = self.cpy_unet.conv_in(similar_imgs[1])
        


        # print("hint block process done===============",torch.cuda.memory_summary())

        
        for idx, down in enumerate(self.cpy_unet.down_blocks):
            if hasattr(down, 'attentions'):

                # preparing similar images from the downblock
                similar_img_one = down(similar_img_one, self.zero_ctrl_unet_temb, encoder_hidden_states=ctrl_txt_condn)
                similar_img_two =  down(similar_img_two, self.zero_ctrl_unet_temb, encoder_hidden_states=ctrl_txt_condn)

                #getting refrence image output of the downblock
                ctrl_unet_out = down(ctrl_unet_out, ctrl_unet_temb, encoder_hidden_states=ctrl_txt_condn)                
            else:
                ctrl_unet_out = down(ctrl_unet_out, ctrl_unet_temb)
                similar_img_one = down(similar_img_one, self.zero_ctrl_unet_temb)
                similar_img_two =  down(similar_img_two, self.zero_ctrl_unet_temb)

            
            if isinstance(ctrl_unet_out, tuple):
                ctrl_unet_out, res_samples = ctrl_unet_out
                if isinstance(similar_img_one,tuple) and isinstance(similar_img_two,tuple):
                    similar_img_one,sim_res_one = similar_img_one
                    similar_img_two,sim_res_two = similar_img_two

                # print("Cross attention happening ",ctrl_unet_out.shape," === ",res[0].shape)
                # now getting cross attention value for the image
                res_samples = list(res_samples) # so as I can modify the list to perform the element updation functionx 
                res_samples_len = len(res_samples)
        
                # to tackle the exploding gradients issue, we will use cross attention to the control unet output and the similar images
                # print("Cross attention happening ",ctrl_unet_out.shape," === ",similar_img_one.shape,similar_img_two.shape)
                attn1 = self.cross_attn[0][idx](ctrl_unet_out,similar_img_one)
                attn2 = self.cross_attn[1][idx](ctrl_unet_out,similar_img_two)

                # print("ctrl_unet_out min max mean",ctrl_unet_out.min().item(),ctrl_unet_out.max().item(),ctrl_unet_out.mean().item())
                ctrl_unet_out = (attn1 + attn2)
                ctrl_unet_out = F.layer_norm(ctrl_unet_out, ctrl_unet_out.shape[1:])  # Layer normalization

                # print("Cross attentiond done",ctrl_unet_out.min().item(),ctrl_unet_out.max().item(),ctrl_unet_out.mean().item())
                res_attn_one_0 = self.cross_attn[0][idx](res_samples[0],sim_res_one[0])
                res_attn_one_1 = self.cross_attn[0][idx](res_samples[1],sim_res_one[1])

                res_attn_two_0 = self.cross_attn[1][idx](res_samples[0],sim_res_two[0])
                res_attn_two_1 = self.cross_attn[1][idx](res_samples[1],sim_res_two[1])
                
                res_samples[0] = (res_attn_one_0 + res_attn_two_0)
                res_samples[1] = (res_attn_one_1 + res_attn_two_1)

                res_samples[0] = F.layer_norm(res_samples[0], res_samples[0].shape[1:])  # Layer normalization
                res_samples[1] = F.layer_norm(res_samples[1], res_samples[1].shape[1:])  # Layer normalization
                
                if(res_samples_len==3):
                    
                    res_samples_three_0 = self.cross_attn[0][idx](res_samples[2],sim_res_one[2])
                    res_samples_three_1 = self.cross_attn[1][idx](res_samples[2],sim_res_two[2])
                    res_samples[2] = (res_samples_three_0 + res_samples_three_1)
                    res_samples[2] = F.layer_norm(res_samples[2], res_samples[2].shape[1:])
                    res_samples = (res_samples[0],res_samples[1],res_samples[2]) # again converting it into a tuple
                else:
                    res_samples = (res_samples[0],res_samples[1])

                
                 
                # Apply zero convolutions to each residual sample
                processed_samples = []
                for res in res_samples:
                    # Choose appropriate zero conv based on channel dimensions
                    if res.shape[1] in self.down_in_channel_resnet:
                        idx_match = self.down_in_channel_resnet.index(res.shape[1])
                        processed_samples.append(self.ctrl_unet_down_zero_convs_resnet[idx_match](res))
                    else:
                        processed_samples.append(self.ctrl_unet_down_zero_convs_mout[idx](res))
                
                ctrl_down_block_res_samples.extend(processed_samples)
        


        # print("ctrldown block process done===============",torch.cuda.memory_summary())


        # Process midblocks
        if isinstance(ctrl_unet_out, tuple):
            ctrl_unet_out = ctrl_unet_out[0]
        
        if isinstance(org_unet_out, tuple):
            org_unet_out = org_unet_out[0]

        
          # Ensure text condition is on the control UNet device
        ctrl_mid_out = self.cpy_unet.mid_block(ctrl_unet_out, ctrl_unet_temb, encoder_hidden_states=ctrl_txt_condn)
        org_mid_out = self.org_unet.mid_block(org_unet_out, org_unet_temb, encoder_hidden_states=text_condn)
        
        # Apply zero conv to control mid output and add to original mid output
        ctrl_mid_out = self.ctrl_unet_mid_zero_convs(ctrl_mid_out)
        ctrl_mid_out = ctrl_mid_out.to(org_unet_device)
         # for proper gradient flow
        
        # org_mid_out = org_mid_out.to(ctrl_unet_device,non_blocking=True)  # Ensure both are on the same device
        # org_mid_out = org_mid_out.to(ctrl_unet_device)# Ensure both are on the same device
        org_unet_out = org_mid_out + ctrl_mid_out  # putting ctrl_net mid output to the same device as original mid output
        
        # Combine the original and control residual samples
        assert len(down_block_res_samples) == len(ctrl_down_block_res_samples), f"Mismatched{len(down_block_res_samples)}====={len(ctrl_down_block_res_samples)} number of residual samples"
        
        combined_down_block_res_samples = []
        for org_res, ctrl_res in zip(down_block_res_samples, ctrl_down_block_res_samples):
            assert org_res.shape == ctrl_res.shape, f"Shape mismatch: {org_res.shape} vs {ctrl_res.shape}"
            ctrl_res = ctrl_res.to(org_unet_device)
            # org_res = org_res.to(ctrl_unet_device)  # Ensure both are on the same device
            combined_down_block_res_samples.append(org_res + ctrl_res)
    
        # cleaning memory for up blocks
        

    # concatenate the similar image embedding with the text condition
        similar_img_clip_embed = [img.to(org_unet_device) for img in similar_img_clip_embed]  # Ensure similar images are on the same device
        conditioning = torch.cat((text_condn, similar_img_clip_embed[0],similar_img_clip_embed[1]), dim=2)
        conditioning = self.converge(conditioning)  # Ensure conditioning is on the same device as org_unet

        # print("midblock process done===============",torch.cuda.memory_summary())


        # Process through up blocks using diffusers pattern

        # now finally moving all tensors to the org_unet device
      
        for i, upsample_block in enumerate(self.org_unet.up_blocks):
            is_final_block = i == len(self.org_unet.up_blocks) - 1
            
            # Get the appropriate residual samples for this upblock
            if hasattr(upsample_block, 'resnets'):
                num_res_samples = len(upsample_block.resnets)
                res_samples = combined_down_block_res_samples[-num_res_samples:]  #getting last num_res_samples to be given to upblock
                combined_down_block_res_samples = combined_down_block_res_samples[:-num_res_samples] # storing the rest for next upblock
            else:
                # Fallback if structure is unknown
                if combined_down_block_res_samples:
                    res_samples = [combined_down_block_res_samples.pop()]
                else:
                    res_samples = []
            
            # Handle upsample size for non-final blocks
            upsample_size = None
            if not is_final_block and combined_down_block_res_samples:
                upsample_size = combined_down_block_res_samples[-1].shape[2:] # it tells the upblock to upsample to this size
            
            # Pass through upblock with appropriate parameters
            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                org_unet_out = upsample_block(
                    hidden_states=org_unet_out,
                    temb=org_unet_temb,
                    res_hidden_states_tuple=tuple(res_samples),
                    encoder_hidden_states=conditioning,
                    # encoder_hidden_states=text_condn,
                    upsample_size=upsample_size
                )
            else:
                org_unet_out = upsample_block(
                    hidden_states=org_unet_out,
                    temb=org_unet_temb,
                    res_hidden_states_tuple=tuple(res_samples),
                    upsample_size=upsample_size
                )
            
            # print(f"Upsample block {i}: {org_unet_out.shape}")
        
        # Final output processing as to maintain the gradient floe
        org_unet_out = self.org_unet.conv_norm_out(org_unet_out)
        org_unet_out = self.org_unet.conv_act(org_unet_out)
        org_unet_out = self.org_unet.conv_out(org_unet_out)
        
        # print("Final output shape:", org_unet_out.min().item(), org_unet_out.max().item(), org_unet_out.mean().item())
        del ctrl_down_block_res_samples,down_block_res_samples
        # print("up block process done===============",torch.cuda.memory_summary())
        
        # clean up
#        torch.cuda.empty_cache()

        return org_unet_out
