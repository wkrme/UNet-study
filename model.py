import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3x3ReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3ReLU, self).__init__()
        self.conv3x3relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv3x3relu(x)

class MaxPool2x2(nn.Module):
    def __init__(self, in_channels):
        super(MaxPool2x2, self).__init__()
        self.maxpool2x2 = nn.MaxPool2d(in_channels)

    def forward(self, x):
        return self.maxpool2x2(x)

class UpConv2x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv2x2, self).__init__()
        self.up2x2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv3x3relu1 = Conv3x3ReLU(in_channels, out_channels)
        self.conv3x3relu2 = Conv3x3ReLU(out_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.up2x2(x1)

        y_diff = x2.size()[2] - x1.size()[2]
        x_diff = x2.size()[3] - x1.size()[3]

        x2 = x2[:, :, (x_diff // 2):(x2.size()[3] - x_diff // 2), (y_diff // 2):(x2.size()[2] - y_diff // 2)]

        x = torch.cat([x2, x1], dim=1)


        x = self.conv3x3relu1(x)
        x = self.conv3x3relu2(x)

        return x

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv1x1(x)

class DropOut(nn.Module):
    def __init__(self, p):
        super(DropOut, self).__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        return self.dropout(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.input1 = Conv3x3ReLU(1, 64)
        self.conv3x3relu2 = Conv3x3ReLU(64, 64)
        self.maxpool3 = MaxPool2x2(2)
        self.conv3x3relu4 = Conv3x3ReLU(64, 128)
        self.conv3x3relu5 = Conv3x3ReLU(128, 128)
        self.maxpool6 = MaxPool2x2(2)
        self.conv3x3relu7 = Conv3x3ReLU(128, 256)
        self.conv3x3relu8 = Conv3x3ReLU(256, 256)
        self.maxpool9 = MaxPool2x2(2)
        self.conv3x3relu10 = Conv3x3ReLU(256, 512)
        self.conv3x3relu11 = Conv3x3ReLU(512, 512)
        self.maxpool12 = MaxPool2x2(2)
        self.conv3x3relu13 = Conv3x3ReLU(512, 1024)
        self.conv3x3relu14 = Conv3x3ReLU(1024, 1024)
        self.dropout = DropOut(0.5)
        self.upconv15 = UpConv2x2(1024, 512)
        self.upconv16 = UpConv2x2(512, 256)
        self.upconv17 = UpConv2x2(256, 128)
        self.upconv18 = UpConv2x2(128, 64)
        self.conv1x1 = Conv1x1(64, 1)

    def forward(self, x):
        x1 = self.input1(x)
        x2 = self.conv3x3relu2(x1) # skip-connection4
        x3 = self.maxpool3(x2)
        x4 = self.conv3x3relu4(x3)
        x5 = self.conv3x3relu5(x4) # skip-connection3
        x6 = self.maxpool6(x5)
        x7 = self.conv3x3relu7(x6)
        x8 = self.conv3x3relu8(x7) # skip-connection2
        x9 = self.maxpool9(x8)
        x10 = self.conv3x3relu10(x9)
        x11 = self.conv3x3relu11(x10) # skip-connection1
        x12 = self.maxpool12(x11)
        x13 = self.conv3x3relu13(x12)
        x14 = self.conv3x3relu14(x13)
        x15 = self.dropout(x14)

        x = self.upconv15(x15, x11)
        x = self.upconv16(x, x7)
        x = self.upconv17(x, x4)
        x = self.upconv18(x, x1)
        output = self.conv1x1(x)

        return output

if __name__ == "__main__":
    model = UNet()
    x = torch.randn(1, 1, 572, 572)
    y = model(x)

    print(y.shape)