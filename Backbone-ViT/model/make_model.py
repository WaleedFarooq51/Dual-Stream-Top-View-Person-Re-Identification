from model.vision_transformer import ViT
import torch
import torch.nn as nn

# L2 norm
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
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class build_vision_transformer(nn.Module):
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer, self).__init__()
        self.in_planes = 768
        
        self.base = ViT(img_size=[cfg.H,cfg.W],
                        stride_size=cfg.STRIDE_SIZE,
                        drop_path_rate=cfg.DROP_PATH,
                        drop_rate=cfg.DROP_OUT,
                        attn_drop_rate=cfg.ATT_DROP_RATE)
     
        self.base.load_param(cfg.PRETRAIN_PATH)
        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)

    def forward(self, fused_features):

        fused_features = self.base(fused_features)

        fused_bn_features = self.bottleneck(fused_features)
        
        if self.training:
            cls_score = self.classifier(fused_bn_features)

            return cls_score, fused_bn_features

        else:
            return self.l2norm(fused_bn_features)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        model_dict = self.state_dict()

        # print("=== Keys in pre-trained weights ===")
        # for key in param_dict.keys():
        #     print(key)
        # print("\n=== Keys in model state_dict ===")
        # for key in model_dict.keys():
        #     print(key)

        for i in param_dict:
            key = i.replace('module.', '')
            param_shape = param_dict[i].shape
            #print(f"Layer: {key}, Pre-trained shape: {param_shape}")

            if key in self.state_dict():
                param_shape = param_dict[i].shape
                model_shape = self.state_dict()[key].shape
                # print(f"Layer: {key} | Shape: {param_shape}")
                # if param_shape != model_shape:
                #     print(f"Size mismatch at layer {key}: checkpoint shape {param_shape}, model shape {model_shape}")
                # else:
                #     self.state_dict()[key].copy_(param_dict[i])       
                
                if key in self.state_dict() and self.state_dict()[key].shape == param_dict[i].shape:
                    self.state_dict()[key].copy_(param_dict[i])
                    #print(f"Layer: {key} | Shape: {param_shape}")
                else:
                    print(f"Skipping layer {key} due to size mismatch or missing layer")
                    
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))