import torch
import torch.nn as  nn
import torch.nn.functional as F
import torchvision

class DeepVision_VGG16(nn.Module):
    
    """
    * Original paper https://arxiv.org/abs/1611.09932
    * Conv1 - Conv 4 is based on VGG-16 (the resulting feature map has receptive field of 92 x 92 with stride 8)
    * Asymmetric two-stream architecture: Conv 5 (G-Stream) and Conv 6 (P-Stream)
    
    We will name the novel CNN as "DeepVision", to honor the course. The variable names here follow the original paper:
        1. M: number of classes (stanford cars dataset has 196 classes)
        2. k: number of discriminative patch detectors per class (in the paper, k=10)

    """
    def __init__(self, k = 10, M = 196): #stanford cars has 196 classes
        super(DeepVision_VGG16, self).__init__()
        self.k = k
        self.M = M
        
        # use VGG-16 as in the original paper
        # suitable due to small size and stride (of conv filters and pooling kernels)
        vgg16_features = torchvision.models.vgg16_bn(pretrained=True).features
        conv4_3 = nn.Sequential(*list(vgg16_features.children())[:-11])
        self.conv4_3 = conv4_3
        
        # G-stream
        conv5 = nn.Sequential(*list(vgg16_features.children())[-11:])
        self.conv5 = conv5
        self.cls5 = nn.Sequential(
            nn.Conv2d(512, 200, kernel_size=1),
            nn.BatchNorm2d(200),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        
        # P-stream
        conv6 = nn.Conv2d(512, k * M, kernel_size = 1) #(1 x 1) convolution, k*M filters are required
        pool6 = nn.MaxPool2d((56, 56), stride = (56, 56), return_indices = True) # keep the indices for visualization purpose
        self.conv6 = conv6
        self.pool6 = pool6
        self.cls6 = nn.Sequential(
            nn.Conv2d(k * M, M, kernel_size=1),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        
        # side branch: conv filter supervision
        self.pool_crosschannel = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)
        
        # TODO:
        # add ReLU
        # add non-random initialization

    def forward(self, x):
        batch_size = x.size(0)
        
        # get feature map (output of conv4_3)
        feature_map = self.conv4_3(x)
        
        # G-stream
        out_g = self.conv5(feature_map)
        out_g = self.cls5(out_g)
        out_g = out_g.view(batch_size, -1)
        
        # P-stream
        out_p = self.conv6(feature_map)
        out_p, indices = self.pool6(out_p)
        temp_p = out_p
        out_p = self.cls6(out_p)
        out_p = out_p.view(batch_size, -1)
        
        # side branch: conv filter supervision
        temp_p = temp_p.view(batch_size, -1, self.k * self.M)
        out_s = self.pool_crosschannel(temp_p)
        out_s = out_s.view(batch_size, -1)
        
        return out_g, out_p, out_s, indices
    
if __name__ == '__main__':
    # Testing
    x = torch.randn(10, 3, 224, 224) # 10 batch, 3 channels
    net = DeepVision_VGG16()
    y = net(x)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)