import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------
# Implementation of a classic neural network
# ------------------------------------------

class NeuralNetwork(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_dim = input_shape
        self.network = nn.Sequential(
            # Fully connected layers
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            
            # Output layer for binary classification
            nn.Linear(16, 1)  # One output will give logits for win which allow to get win probability percentage
        )
    
    def forward(self, x):
        x = self.network(x)
        logits = torch.sigmoid(x)
        return logits


# ------------------------------------------
# Implementation of a SkipNet
# ------------------------------------------

class SkipBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SkipBlock, self).__init__()
        # Transformation part: two fully-connected layers with a non-linearity
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        # Gate: produces a scalar weight for each example
        self.gate_fc = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()
        
        # If the input and output sizes differ, we add a projection to enable the skip connection.
        if in_features != out_features:
            self.skip_proj = nn.Linear(in_features, out_features)
        else:
            self.skip_proj = None

    def forward(self, x):
        # Compute the gate value (between 0 and 1)
        gate = self.sigmoid(self.gate_fc(x))
        
        # Compute transformed output
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        
        # If dimensions differ, project the input for the skip path
        skip = self.skip_proj(x) if self.skip_proj is not None else x
        
        # Combine transformed output and skip connection weighted by the gate
        # When gate is close to 1, we use the transformed output; when it's close to 0, we use the input.
        out = gate * h + (1 - gate) * skip
        return out, gate

class SkipNetMLP(nn.Module):
    def __init__(self, input_dim):
        super(SkipNetMLP, self).__init__()
        self.skip_block1 = SkipBlock(input_dim, 256, 256)
        self.skip_block2 = SkipBlock(256, 128, 128)
        self.skip_block3 = SkipBlock(128, 64, 64)
        self.skip_block4 = SkipBlock(64, 32, 32)
        self.skip_block5 = SkipBlock(32, 16, 16)
        
        # Final classification layer (no skip here)
        self.fc_out = nn.Linear(16, 1)
        
    def forward(self, x):
        # Pass through the skip blocks
        x, gate1 = self.skip_block1(x)
        x, gate2 = self.skip_block2(x)
        x, gate3 = self.skip_block3(x)
        x, gate4 = self.skip_block4(x)
        x, gate5 = self.skip_block5(x)
        
        # Final output layer with sigmoid for binary classification probability
        out = torch.sigmoid(self.fc_out(x))
        return out, (gate1, gate2, gate3, gate4, gate5)
    

# ------------------------------------------
# Implementation of a ResNet
# ------------------------------------------

class BasicBlock(nn.Module):
    """
    A residual block for tabular data.
    It uses two fully connected layers (with BatchNorm and ReLU)
    and a skip connection.
    """
    def __init__(self, in_features, out_features, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Shortcut projection if dimensions differ
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetTabular(nn.Module):
    """
    A ResNet-style MLP for tabular data.
    It starts with an initial projection from the input features,
    then applies several residual blocks, and finally produces the output.
    """
    def __init__(self, input_dim, num_classes=2, dropout_rate=0.3):
        super(ResNetTabular, self).__init__()
        self.in_features = 64
        self.fc_initial = nn.Linear(input_dim, self.in_features)
        self.bn_initial = nn.BatchNorm1d(self.in_features)
        self.relu = nn.ReLU(inplace=True)
        
        # Build layers: adjust the number of blocks and hidden sizes as needed
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks=2, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks=2, dropout_rate=dropout_rate)
        
        # Final classification layer. For binary classification, you can output one value
        # with a sigmoid, or for multi-class use num_classes outputs with softmax.
        self.fc_out = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, out_features, num_blocks, dropout_rate):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_features, out_features, dropout_rate))
            self.in_features = out_features  # update input dimension for the next block
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial projection
        out = self.fc_initial(x)
        out = self.bn_initial(out)
        out = self.relu(out)

        # Pass through residual layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Final output layer with sigmoid for binary classification
        out = self.fc_out(out)
        out = torch.sigmoid(out)
        return out
