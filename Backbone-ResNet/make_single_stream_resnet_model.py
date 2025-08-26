from torch.nn import init
from resnet import resnet50
import torch
import torch.nn as nn
import os


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
    
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

class stem_block_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(stem_block_resnet, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)

        return x
    
    def load_param(self, checkpoint_path):

        if os.path.isfile(checkpoint_path):
            print('==> loading checkpoint {} for Visible module'.format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)

        filtered_dict = { k.replace('stem_block_resnet.', ''): v 
            for k, v in checkpoint.items() 
            if k.startswith('stem_block_resnet.visible') and k.replace('stem_block_resnet.', '') in self.state_dict() }
        
        try:
            self.load_state_dict(filtered_dict, strict=False)
            print(f"Successfully loaded params into stem_block_resnet.")

        except:
            print(f"[stem_block_resnet] Missing keys: {self.load_state_dict(filtered_dict, strict=False).missing_keys}")

class residual_block_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(residual_block_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x
    
    def load_param(self, checkpoint_path):
        
        if os.path.isfile(checkpoint_path):
            print('==> loading checkpoint {} for residual_block_resnet module'.format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)

        filtered_dict = { k.replace('residual_block_resnet.', ''): v 
            for k, v in checkpoint.items() 
            if k.startswith('residual_block_resnet.base') and k.replace('residual_block_resnet.', '') in self.state_dict() }
                
        try:
            self.load_state_dict(filtered_dict, strict=False)
            print(f"Successfully loaded params into residual_block_resnet module.")

        except:
            print(f"[residual_block_resnet_module] Missing keys: {self.load_state_dict(filtered_dict, strict=False).missing_keys}")
    
class single_stream_net(nn.Module):
    def __init__(self,  class_num, arch='resnet50'):
        super(single_stream_net, self).__init__()

        self.stem_block_resnet = stem_block_resnet(arch=arch)
       
        self.residual_block_resnet = residual_block_resnet(arch=arch)
        
        pool_dim = 2048      #defualt feature dimension when using resnet backbone

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, features):

        if self.training:
            x= self.stem_block_resnet(features)
      
            x= self.residual_block_resnet(x)
            
            pool_features= self.avgpool(x)
            pool_features = pool_features.view(pool_features.size(0), pool_features.size(1))
            
            bn_features = self.bottleneck(pool_features)

            return pool_features, self.classifier(bn_features)
    
        else:
            feats= self.stem_block_resnet(features)

            feats= self.residual_block_resnet(feats)
            
            pool_features= self.avgpool(feats)
            pool_features = pool_features.view(pool_features.size(0), pool_features.size(1))
            
            bn_features = self.bottleneck(pool_features)

            return self.l2norm(pool_features), self.l2norm(bn_features)
        

   