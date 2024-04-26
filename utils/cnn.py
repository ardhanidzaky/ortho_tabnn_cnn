import timm
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, model_key, fc_input_num, num_classes=3, in_channel=3):
        super(CNN, self).__init__()
        
        self.backbone = timm.create_model(model_key, in_chans=in_channel, pretrained=True, features_only=True, drop_rate=0.5)        
        self.max_pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.num_classes = num_classes

    def forward(self, x):
        x = self.backbone(x)
        x = self.max_pool(x[-1])
        
        # Transformer-specific
        fc_input_num = self._calculate_fc(x)
        x = self.flatten(x) 
        fc = nn.Linear(fc_input_num, self.num_classes).to('mps')
        x = fc(x)
        
        return x
    
    def _calculate_fc(self, x):
        return x.size(1) * x.size(2) * x.size(3)