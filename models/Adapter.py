
import torch.nn as nn
#this is the original class in the original repo
'''class Adapter(nn.Module):

    def __init__(self,
                 clip_model,
                 target,
                 ):

        super(Adapter, self).__init__()

        input_sizes = clip_model.token_c

        for i,input_size in enumerate(input_sizes):
            self.add_module("{}_adapter".format(i), nn.Sequential(nn.Conv2d(input_size, target, 1, 1)))


    def forward(self, tokens):
        vision_features=[]
        for i,token in enumerate(tokens):
            vision_feature=getattr(self,'{}_adapter'.format(i))(token).contiguous().permute(0, 2, 3, 1)
            vision_feature = vision_feature / vision_feature.norm(dim=-1, keepdim=True)
            vision_features.append(vision_feature)
        return vision_features'''

#This is the modified class

class Adapter(nn.Module):
    def __init__(self, clip_model, target):
        super(Adapter, self).__init__()
        input_sizes = clip_model.token_c
        self.adapters = nn.ModuleList()

        for i, input_size in enumerate(input_sizes):
            self.adapters.append(
                nn.Sequential(
                    nn.Conv2d(input_size, target, kernel_size=1),
                    nn.BatchNorm2d(target),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                )
            )

    def forward(self, tokens):
        vision_features = []
        for i, token in enumerate(tokens):
            # Apply adapter
            vision_feature = self.adapters[i](token)
            # Normalize and permute
            vision_feature = vision_feature / vision_feature.norm(dim=1, keepdim=True)
            vision_feature = vision_feature.permute(0, 2, 3, 1)  # Match expected downstream format
            vision_features.append(vision_feature)
        return vision_features
