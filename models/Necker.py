
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
#original class implementation
# class Necker(nn.Module):

#     def __init__(self,
#                   clip_model
#                  ):
#         super(Necker, self).__init__()
#         self.clip_model=clip_model
#         target = max(self.clip_model.token_size)
#         for i,size in enumerate(self.clip_model.token_size):
#             self.add_module("{}_upsample".format(i),
#                                 nn.UpsamplingBilinear2d(scale_factor=target/size))


#     @torch.no_grad()
#     def forward(self, tokens):
#         align_features=[]
#         for i,token in enumerate(tokens):
#             if len(token.shape) == 3:
#                 B, N, C=token.shape
#                 token = token[:, 1:, :]
#                 token=token.view((B,int(math.sqrt(N-1)),int(math.sqrt(N-1)),C)).permute(0, 3, 1, 2)
#             align_features.append(getattr(self, "{}_upsample".format(i))(token))
#         return align_features


#This class with the changed adapter produced the AUROC of 0.9177
# class Necker(nn.Module):
#     def __init__(self, clip_model):
#         super(Necker, self).__init__()
#         self.clip_model = clip_model
#         target = max(self.clip_model.token_size)
        
#         # Layer weights for adaptive feature fusion
#         self.layer_weights = nn.Parameter(torch.ones(len(self.clip_model.token_size)))
#         self.softmax = nn.Softmax(dim=0)
        
#         # Enhanced modules for each layer
#         for i, size in enumerate(self.clip_model.token_size):
#             # Upsample module
#             self.add_module(f"{i}_upsample", 
#                 nn.UpsamplingBilinear2d(scale_factor=target/size))
            
#             # Layer normalization
#             self.add_module(f"{i}_norm",
#                 nn.LayerNorm(self.clip_model.token_c[i]))
            
#             # Self-attention
#             self.add_module(f"{i}_attention",
#                 nn.MultiheadAttention(
#                     embed_dim=self.clip_model.token_c[i],
#                     num_heads=8,
#                     dropout=0.1,
#                     batch_first=True
#                 ))
            
#             # Dropout
#             self.add_module(f"{i}_dropout",
#                 nn.Dropout(0.1))

#     @torch.no_grad()
#     def forward(self, tokens):
#         align_features = []
#         weights = self.softmax(self.layer_weights)
        
#         for i, token in enumerate(tokens):
#             # Apply layer norm
#             token = getattr(self, f"{i}_norm")(token)
            
#             if len(token.shape) == 3:
#                 B, N, C = token.shape
                
#                 # Self-attention
#                 token = token[:, 1:, :]
#                 attn_out, _ = getattr(self, f"{i}_attention")(token, token, token)
                
#                 # Reshape and permute
#                 token = attn_out.view((B, int(math.sqrt(N-1)), 
#                                      int(math.sqrt(N-1)), C)).permute(0, 3, 1, 2)
                
#                 # Dropout
#                 token = getattr(self, f"{i}_dropout")(token)
            
#             # Upsample and weight
#             upsampled = getattr(self, f"{i}_upsample")(token)
#             align_features.append(upsampled * weights[i])
            
#         return align_features


class RegionAwareModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=7, padding=3)
        self.scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Spatial attention map
        spatial_map = torch.mean(x, dim=1, keepdim=True)
        attention = torch.sigmoid(self.conv1(spatial_map))
        return x * (attention * self.scale)

class Necker(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        target = max(self.clip_model.token_size)
        
        for i, size in enumerate(self.clip_model.token_size):
            # Region-aware module
            self.add_module(f"{i}_region", 
                          RegionAwareModule(self.clip_model.token_c[i]))
            
            # Basic upsampling with refinement
            self.add_module(f"{i}_upsample", 
                nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=target/size),
                    nn.Conv2d(self.clip_model.token_c[i], 
                             self.clip_model.token_c[i], 1),
                    nn.BatchNorm2d(self.clip_model.token_c[i]),
                    nn.ReLU(inplace=True)
                ))
    
    @torch.no_grad()
    def forward(self, tokens):
        align_features = []
        
        for i, token in enumerate(tokens):
            if len(token.shape) == 3:
                B, N, C = token.shape
                token = token[:, 1:, :]
                token = token.view((B, int(math.sqrt(N-1)), 
                                  int(math.sqrt(N-1)), C)).permute(0, 3, 1, 2)
                
                # Region-aware processing
                token = getattr(self, f"{i}_region")(token)
                # Upsampling
                token = getattr(self, f"{i}_upsample")(token)
                
            align_features.append(token)
        
        return align_features