
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




class FeatureRefinement(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return F.relu(x + residual)

class Necker(nn.Module):
    def __init__(self, clip_model):
        super(Necker, self).__init__()
        self.clip_model = clip_model
        target = max(self.clip_model.token_size)
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        self.layer_weights = nn.Parameter(torch.ones(len(self.clip_model.token_size)))
        
        for i, size in enumerate(self.clip_model.token_size):
            # Enhanced upsampling
            self.add_module(f"{i}_upsample", nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=target/size),
                nn.Conv2d(self.clip_model.token_c[i], self.clip_model.token_c[i], 1),
                nn.BatchNorm2d(self.clip_model.token_c[i]),
                nn.ReLU()
            ))
            
            # Layer normalization
            self.add_module(f"{i}_norm", nn.LayerNorm(self.clip_model.token_c[i]))
            
            # Feature refinement
            self.add_module(f"{i}_refine", FeatureRefinement(self.clip_model.token_c[i]))
            
            # Channel attention weights
            self.add_module(f"{i}_channel_weights", nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.clip_model.token_c[i], self.clip_model.token_c[i], 1),
                nn.Sigmoid()
            ))

    @torch.no_grad()
    def forward(self, tokens):
        align_features = []
        weights = F.softmax(self.layer_weights / self.temperature, dim=0)
        
        prev_feature = None
        for i, token in enumerate(tokens):
            if len(token.shape) == 3:
                B, N, C = token.shape
                # Layer norm
                token = getattr(self, f"{i}_norm")(token)
                # Remove CLS and reshape
                token = token[:, 1:, :]
                token = token.view((B, int(math.sqrt(N-1)), 
                                 int(math.sqrt(N-1)), C)).permute(0, 3, 1, 2)
                
                # Feature refinement
                token = getattr(self, f"{i}_refine")(token)
                
                # Channel attention
                channel_weights = getattr(self, f"{i}_channel_weights")(token)
                token = token * channel_weights
                
                # Skip connection with previous feature if available
                if prev_feature is not None and prev_feature.shape[2:] == token.shape[2:]:
                    token = token + prev_feature
                
                # Upsample
                token = getattr(self, f"{i}_upsample")(token)
                prev_feature = token
                
            align_features.append(token * weights[i])
        
        return align_features