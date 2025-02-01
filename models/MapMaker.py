
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

#original code as in the paper
# class MapMaker(nn.Module):

#     def __init__(self,image_size):

#         super(MapMaker, self).__init__()
#         self.image_size = image_size

#     #this is the original code as in the paper
#     def forward(self, vision_adapter_features,propmt_adapter_features):
#         anomaly_maps=[]

#         for i,vision_adapter_feature in enumerate(vision_adapter_features):
#             B, H, W, C = vision_adapter_feature.shape
            
#             #this commented is the original
#             #anomaly_map = (vision_adapter_feature.view((B, H * W, C)) @ propmt_adapter_features).contiguous().view(
#             anomaly_map = (vision_adapter_feature.reshape((B, H * W, C)) @ propmt_adapter_features).contiguous().view(

#                 (B, H, W, -1)).permute(0, 3, 1, 2)

#             anomaly_maps.append(anomaly_map)

#         anomaly_map = torch.stack(anomaly_maps, dim=0).mean(dim=0)
#         anomaly_map = F.interpolate(anomaly_map, (self.image_size, self.image_size), mode='bilinear', align_corners=True)
#         return torch.softmax(anomaly_map, dim=1)
    




    # this was the updated code to run the RN101 model without adapter
    # def forward(self, vision_adapter_feature, prompt_adapter_features):
    #     if isinstance(vision_adapter_feature, list):
    #     # Concatenate along channel dimension
    #         vision_adapter_feature = torch.cat(vision_adapter_feature, dim=1)
    
    # # 1. Get dimensions
    #     B, C, H, W = vision_adapter_feature.shape  # [8, 3840, 56, 56]
    
    # # 2. Reshape vision features
    #     vision_features = vision_adapter_feature.permute(0, 2, 3, 1)  # [8, 56, 56, 3840]
    #     vision_features = vision_features.reshape(B, H * W, C)  # [8, 3136, 3840]
    
    # # 3. Project prompt features to match C dimension
    #     prompt_features = prompt_adapter_features  # [512, 2]
    #     prompt_features = prompt_features.permute(1, 0)  # [2, 512]
    #     prompt_features = prompt_features.unsqueeze(0).expand(B, -1, -1)  # [8, 2, 512]
    
    # # 4. Linear projection to match dimensions
    #     if not hasattr(self, 'projection'):
    #         self.projection = nn.Linear(512, C).to(vision_adapter_feature.device)
    #     prompt_features = self.projection(prompt_features)  # [8, 2, 3840]
    #     prompt_features = prompt_features.permute(0, 2, 1)  # [8, 3840, 2]
    
    # # 5. Matrix multiplication
    #     anomaly_map = torch.matmul(vision_features, prompt_features)  # [8, 3136, 2]
    #     anomaly_map = anomaly_map.view(B, H, W, 2).permute(0, 3, 1, 2)  # [8, 2, 56, 56]
    
    # # 6. Final processing
    #     anomaly_map = F.interpolate(anomaly_map, (self.image_size, self.image_size), mode='bilinear', align_corners=True)
    #     return torch.softmax(anomaly_map, dim=1)


#to run the model_name: ViT-L-14, layers_out: [12, 18, 24] without the adapter
# class MapMaker(nn.Module):
#     def __init__(self, image_size):
#         super(MapMaker, self).__init__()
#         self.image_size = image_size
#         self.projection = nn.Linear(768, 3072)

#     def forward(self, vision_adapter_features, prompt_features):
#         anomaly_maps = []
        
#         for vision_feature in vision_adapter_features:
#             # Handle 3D tensor input
#             if vision_feature.dim() == 3:
#                 B, C, HW = vision_feature.shape
#                 H = W = int(math.sqrt(HW))  # Assuming square feature map
#                 vision_feature = vision_feature.reshape(B, C, H, W)
            
#             # Now process 4D tensor
#             B, C, H, W = vision_feature.shape  # [8,3072,16,16]
            
#             # Reshape vision features
#             vision_features = vision_feature.permute(0, 2, 3, 1).reshape(B, H * W, C)   
#             print(f"vision_features shape after reshaping: {vision_features.shape}")
#               # Debug statement

#             # Project prompt features
#             #prompt_features = prompt_features.t()  # [2,768]
#             projected_prompts = self.projection(prompt_features)  # [2,3072]
#             projected_prompts = projected_prompts.t()  # [3072,2]
            
#             # Matrix multiply
#             anomaly_map = torch.matmul(vision_features, projected_prompts)  # shape: [B, H*W, 2]
#             anomaly_map = anomaly_map.transpose(1, 2).view(B, projected_prompts.shape[1], H, W) 
#             print(f"anomaly_map shape after matmul and reshaping: {anomaly_map.shape}")
            
#             anomaly_maps.append(anomaly_map)
        
#         anomaly_map = torch.stack(anomaly_maps, dim=0).mean(dim=0)
#         anomaly_map = F.interpolate(anomaly_map, (self.image_size, self.image_size), mode='bilinear', align_corners=True)
#         return torch.softmax(anomaly_map, dim=1)




import torch
import torch.nn as nn
import torch.nn.functional as F

class MapMaker(nn.Module):
    def __init__(self, image_size, vision_channels):
        super(MapMaker, self).__init__()
        self.image_size = image_size
        self.projection = None  # Will be initialized in forward
    
    def forward(self, vision_features, prompt_features):
        while isinstance(vision_features, list):
            vision_features = vision_features[0]
            
        B, C, H, W = vision_features.shape
        
        # Initialize projection layer if not exists or if channels changed
        if self.projection is None or self.projection.out_features != C:
            self.projection = nn.Linear(768, C).to(vision_features.device)
            
        vision_feats = vision_features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        #print(f"MapMaker - vision_feats reshaped: {vision_feats.shape}")
        
        if prompt_features.shape[1] == 2:
            prompt_features = prompt_features.t()
        
        proj_prompts = self.projection(prompt_features)  # [2, C]
        proj_prompts = proj_prompts.t()  # [C, 2]
        #print(f"MapMaker - projected prompts shape: {proj_prompts.shape}")
        
        anomaly_map = torch.matmul(vision_feats, proj_prompts)  # [B, H*W, 2]
        anomaly_map = anomaly_map.transpose(1, 2).view(B, 2, H, W)
        
        anomaly_map = F.interpolate(anomaly_map, (self.image_size, self.image_size), 
                                    mode='bilinear', align_corners=True)
        return torch.softmax(anomaly_map, dim=1)