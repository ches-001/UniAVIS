import torch
import torch.nn as nn
from torchvision.models import resnet
from typing import *


class ResNetBackBone(resnet.ResNet):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        block: Union[str, Type]=resnet.BasicBlock, 
        block_layers: Optional[Iterable[int]]=None
    ):
        if isinstance(block, str):
            block = getattr(resnet, block)
        super(ResNetBackBone, self).__init__(block=block, layers=block_layers or [3, 4, 6, 3])

        self.in_channels  = in_channels  
        self.out_channels = out_channels
        self.conv1        = nn.Conv2d(
            self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        if block == resnet.BasicBlock:
            self.fmaps_channels = (64, 128, 256, 512)
        elif block == resnet.Bottleneck:
            self.fmaps_channels = (256, 512, 1024, 2048)

        self.fmap_proj_modules = self._create_fmap_channel_projectors()

        #delete unwanted layer(s)
        del self.fc, self.avgpool

    def _create_fmap_channel_projectors(self) -> nn.ModuleList:
        return nn.ModuleList([
            nn.Conv2d(
                in_channels, 
                self.out_channels, 
                kernel_size=(1, 1), 
                stride=(1, 1)
            ) for in_channels in self.fmaps_channels
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Input
        --------------------------------
        :x: (N, C, H, W) or (N*V, C, H, W), batch of input images (where V is number of views)

        Returns
        --------------------------------
        :multiscale_fmaps: List[(N, out_C, H_fmap, W_fmap)], list of feature maps of various spatial resolutions form
                        different levels
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        fmap1 = self.layer1(x)
        fmap2 = self.layer2(fmap1)
        fmap3 = self.layer3(fmap2)
        fmap4 = self.layer4(fmap3)

        multiscale_fmaps = [self.fmap_proj_modules[i](fmap) for i, fmap in enumerate([fmap1, fmap2, fmap3, fmap4])]
        return multiscale_fmaps