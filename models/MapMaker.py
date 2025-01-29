
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MapMaker(nn.Module):

    def __init__(self,image_size):

        super(MapMaker, self).__init__()
        self.image_size = image_size


    # def forward(self, vision_adapter_features,propmt_adapter_features):
    #     anomaly_maps=[]

    #     for i,vision_adapter_feature in enumerate(vision_adapter_features):
    #         B, H, W, C = vision_adapter_feature.shape
            
    #         #this commented is the original
    #         #anomaly_map = (vision_adapter_feature.view((B, H * W, C)) @ propmt_adapter_features).contiguous().view(
    #         anomaly_map = (vision_adapter_feature.reshape((B, H * W, C)) @ propmt_adapter_features).contiguous().view(

    #             (B, H, W, -1)).permute(0, 3, 1, 2)

    #         anomaly_maps.append(anomaly_map)

    #     anomaly_map = torch.stack(anomaly_maps, dim=0).mean(dim=0)
    #     anomaly_map = F.interpolate(anomaly_map, (self.image_size, self.image_size), mode='bilinear', align_corners=True)
    #     return torch.softmax(anomaly_map, dim=1)
    def forward(self, vision_adapter_feature, prompt_adapter_features):
        if isinstance(vision_adapter_feature, list):
        # Concatenate along channel dimension
            vision_adapter_feature = torch.cat(vision_adapter_feature, dim=1)
    
    # 1. Get dimensions
        B, C, H, W = vision_adapter_feature.shape  # [8, 3840, 56, 56]
    
    # 2. Reshape vision features
        vision_features = vision_adapter_feature.permute(0, 2, 3, 1)  # [8, 56, 56, 3840]
        vision_features = vision_features.reshape(B, H * W, C)  # [8, 3136, 3840]
    
    # 3. Project prompt features to match C dimension
        prompt_features = prompt_adapter_features  # [512, 2]
        prompt_features = prompt_features.permute(1, 0)  # [2, 512]
        prompt_features = prompt_features.unsqueeze(0).expand(B, -1, -1)  # [8, 2, 512]
    
    # 4. Linear projection to match dimensions
        if not hasattr(self, 'projection'):
            self.projection = nn.Linear(512, C).to(vision_adapter_feature.device)
        prompt_features = self.projection(prompt_features)  # [8, 2, 3840]
        prompt_features = prompt_features.permute(0, 2, 1)  # [8, 3840, 2]
    
    # 5. Matrix multiplication
        anomaly_map = torch.matmul(vision_features, prompt_features)  # [8, 3136, 2]
        anomaly_map = anomaly_map.view(B, H, W, 2).permute(0, 3, 1, 2)  # [8, 2, 56, 56]
    
    # 6. Final processing
        anomaly_map = F.interpolate(anomaly_map, (self.image_size, self.image_size), mode='bilinear', align_corners=True)
        return torch.softmax(anomaly_map, dim=1)