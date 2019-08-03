import torch
import torch.nn as  nn
import torch.nn.functional as F
import torchvision

class DFL_VGG16(nn.Module):
    """
    * Original paper https://arxiv.org/abs/1611.09932
    * Conv1 - Conv 4 is based on VGG-16 (the resulting feature map has receptive field of 92 x 92 with stride 8)
    * Asymmetric two-stream architecture: Conv 5 (G-Stream) and Conv 6 (P-Stream)

    The variable names here follow the original paper:
        1. M: number of classes (stanford cars dataset has 196 classes)
        2. k: number of discriminative patch detectors per class (in the paper, k=10)

    This Net is mainly used to generate visualization. To get state-of-the-art results, ResNet50 will be preferred.
    """
    def __init__(self, k = 10, nclass = 196):
        super(DFL_VGG16, self).__init__()
        self.k = k
        self.nclass = nclass

        # k channels for one class, nclass is total classes, therefore k * nclass for conv6
        # Details of conv1_x -- conv4_x see VGG-16 paper https://arxiv.org/abs/1409.1556
        vgg16featuremap = torchvision.models.vgg16_bn(pretrained=True).features
        conv1_conv4 = torch.nn.Sequential(*list(vgg16featuremap.children())[:-11]) # output size 56 x 56
        conv5 = torch.nn.Sequential(*list(vgg16featuremap.children())[-11:])
        conv6 = torch.nn.Conv2d(512, k * nclass, kernel_size = 1, stride = 1, padding = 0)
        pool6 = torch.nn.MaxPool2d((56, 56), stride = (56, 56), return_indices = True)

        # Feature extraction root
        self.conv1_conv4 = conv1_conv4

        # G-Stream
        self.conv5 = conv5
        self.cls5 = nn.Sequential(
            nn.Conv2d(512, nclass, kernel_size=1, stride = 1, padding = 0),
            nn.BatchNorm2d(nclass),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)),
            )

        # P-Stream
        self.conv6 = conv6
        self.pool6 = pool6
        self.cls6 = nn.Sequential(
            nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
            nn.AdaptiveAvgPool2d((1,1)),
            )

        # Side-branch
        # https://kite.com/python/docs/torch.nn.AvgPool1d
        # [batch_size, 1, k * M] --> [batch_size, M]
        self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)


    def forward(self, x):
        batchsize = x.size(0)

        # Stem: Feature extraction
        inter4 = self.conv1_conv4(x)

        # G-stream
        x_g = self.conv5(inter4)
        out1 = self.cls5(x_g)
        out1 = out1.view(batchsize, -1)

        # P-stream ,indices is for visualization
        x_p = self.conv6(inter4)
        x_p, indices = self.pool6(x_p)
        inter6 = x_p
        out2 = self.cls6(x_p)
        out2 = out2.view(batchsize, -1)

        # Side-branch
        inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
        out3 = self.cross_channel_pool(inter6)
        out3 = out3.view(batchsize, -1)

        return out1, out2, out3, indices



class DFL_ResNet50(nn.Module):

    """
    We also extend the basic DFL (using VGG16) as presented in the paper to use ResNet50 as feature extractor.
    Modifications and highlights include:
    - Added fully connected layers (conv layer was directly connected to output layer in the quick-and-dirty)
    - Added dropout layer in G-stream (for learning global features)
    - Used batch normalization

    """
    def __init__(self, k = 10, nclass = 196):
        super(DFL_ResNet50, self).__init__()
        self.k = k
        self.nclass = nclass

        # k channels for one class, nclass is total classes, therefore k * nclass for conv6
        resnet50 = torchvision.models.resnet50(pretrained=True)
        # conv1_conv4
        layers_conv1_conv4 = [
        resnet50.conv1,
        resnet50.bn1,
        resnet50.relu,
        resnet50.maxpool,
        ]
        for i in range(3):
            name = 'layer%d' % (i + 1)
            layers_conv1_conv4.append(getattr(resnet50, name))
            conv1_conv4 = torch.nn.Sequential(*layers_conv1_conv4)

        # conv5
        layers_conv5 = []
        layers_conv5.append(getattr(resnet50, 'layer4'))
        conv5 = torch.nn.Sequential(*layers_conv5)

        conv6 = torch.nn.Conv2d(1024, k * nclass, kernel_size = 1, stride = 1, padding = 0)
        pool6 = torch.nn.MaxPool2d((28, 28), stride = (28, 28), return_indices = True)

        # Feature extraction root
        self.conv1_conv4 = conv1_conv4

        # G-Stream
        self.conv5 = conv5
        self.cls5 = nn.Sequential(
            nn.Conv2d(2048, nclass, kernel_size=1, stride = 1, padding = 0),
            #nn.Conv2d(2048, 1024, kernel_size=1, stride = 1, padding = 0),
            nn.BatchNorm2d(nclass),
            #nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)),
            # Add FC units
            #nn.Dropout(),
            )
        #self.fc5 = nn.Linear(1024, nclass)

        # P-Stream
        self.conv6 = conv6
        self.pool6 = pool6
        self.cls6 = nn.Sequential(
            nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
            #nn.Conv2d(k * nclass, k * nclass, kernel_size = 1, stride = 1, padding = 0),
            nn.AdaptiveAvgPool2d((1,1)),
            )
        #self.fc6 = nn.Linear(k * nclass, nclass)

        # Side-branch
        self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)

    def forward(self, x):
        batchsize = x.size(0)

        # Stem: Feature extractor
        inter4 = self.conv1_conv4(x)

        # G-stream
        x_g = self.conv5(inter4)
        out1 = self.cls5(x_g)
        out1 = out1.view(batchsize, -1)
        #out1 = self.fc5(out1)

        # P-stream ,indices are for visualization
        x_p = self.conv6(inter4)
        x_p, indices = self.pool6(x_p)
        inter6 = x_p
        out2 = self.cls6(x_p)
        out2 = out2.view(batchsize, -1)
        #out2 = self.fc6(out2)

        # Side-branch, no FC layers are required here
        inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
        out3 = self.cross_channel_pool(inter6)
        out3 = out3.view(batchsize, -1)

        return out1, out2, out3, indices



class Energy_ResNet50(nn.Module):
    """
    Compute the energy (L2-norm) for each patch of the feature map
    """
    def __init__(self, k = 10, nclass = 196):
        super(Energy_ResNet50, self).__init__()
        self.k = k
        self.nclass = nclass

        # k channels for one class, nclass is total classes, therefore k * nclass for conv6
        resnet50 = torchvision.models.resnet50(pretrained=True)
        # conv1_conv4
        layers_conv1_conv4 = [
        resnet50.conv1,
        resnet50.bn1,
        resnet50.relu,
        resnet50.maxpool,
        ]
        for i in range(3):
            name = 'layer%d' % (i + 1)
            layers_conv1_conv4.append(getattr(resnet50, name))
        conv1_conv4 = torch.nn.Sequential(*layers_conv1_conv4)

        self.conv1_conv4 = conv1_conv4

    def forward(self, x):
        batchsize = x.size(0)

        # Get feature map from the ImageNet pre-trained model
        inter4 = self.conv1_conv4(x) # size 1, 1024, x 28 x 28
        inter4 = inter4.view(-1, 28, 28)
        # compute L2 norm at each patch (across channel)
        center = torch.norm(inter4, dim=0)

        patches = torch.zeros((inter4.shape[0], self.k))
        center = center.view(-1)
        for i in range(self.k):
            idx = torch.argmax(center)
            center[idx] = -1
            offset = idx // 28
            patches[:, i] = inter4[:, offset, (idx - offset * 28)]

        return patches


if __name__ == '__main__':

    input_test = torch.ones(10,3,448,448)

    print('Testing DFL-VGG16...')
    net = DFL_VGG16()
    output_test = net(input_test)
    print(output_test[0].shape)
    print(output_test[1].shape)
    print(output_test[2].shape)
    print('Done testing DFL-VGG16.')

    print('Testing DFL-ResNET...')
    input_test = torch.ones(10,3,448,448)
    net = DFL_ResNet50()
    output_test = net(input_test)
    print(output_test[0].shape)
    print(output_test[1].shape)
    print(output_test[2].shape)

    print('Done testing DFL-ResNET...')

    print('Testing energy computations...')

    input_test = torch.ones(1,3,448,448)
    net = Energy_ResNet50()
    output_test = net(input_test)
    print(output_test.shape)
    print(output_test)
