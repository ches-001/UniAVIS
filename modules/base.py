import torch.nn as nn

class BaseFormer(nn.Module):
    def __init__(self):
        super(BaseFormer, self).__init__()
        self.inference_mode = False

    def set_inference_mode(self):
        self.eval()
        self.inference_mode = True