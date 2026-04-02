# load packages


import torch
import torch.nn.functional as F
import torch.nn as nn


class Deeplob_encoder(nn.Module):
    def __init__(self,in_channels: int = 1, d_model: int = 32,output_dim: int = 32,dropout: float = 0.2):
        super().__init__()
        # self.y_len = y_len
        
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=d_model, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
#             nn.Tanh(),
            nn.BatchNorm2d(d_model),
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(d_model),
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(d_model),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(d_model),
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(d_model),
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(d_model),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(1,10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(d_model),
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(d_model),
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(d_model),
        )
        
        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model * 2, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(d_model * 2),
            nn.Conv2d(in_channels=d_model * 2, out_channels=d_model * 2, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(d_model * 2),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model * 2, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(d_model * 2),
            nn.Conv2d(in_channels=d_model * 2, out_channels=d_model * 2, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(d_model * 2),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=d_model, out_channels=d_model * 2, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(d_model * 2),
        )
        self.fc_fusion = nn.Sequential(
            nn.Linear(d_model * 2 * 3, d_model * 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, output_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout),
        )
        # # lstm layers
        # self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        # self.fc1 = nn.Linear(64, self.y_len)
    @property
    def output_dim(self) -> int:
        return 64*3
    def forward(self, x):

    
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)  
        
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1) ## (B,C_fusion,T,1)
        
# #         x = torch.transpose(x, 1, 2)
        x = x.permute(0, 2, 1, 3) ## (B,T,C_fusion,1)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2])) ## (B,T,C_fusion*1)
        ## 融合映射到output_dim维度
        x = self.fc_fusion(x)

        return x



class Deeplob_encoder_simple(nn.Module):
    def __init__(self,in_channels: int = 1, d_model: int = 32,output_dim: int = 32,dropout: float = 0.2):
        super().__init__()
        # self.y_len = y_len
        
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=d_model, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
#             nn.Tanh(),
            nn.BatchNorm2d(d_model),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            # nn.LeakyReLU(negative_slope=0.01),
            # nn.BatchNorm2d(32),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            # nn.LeakyReLU(negative_slope=0.01),
            # nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(d_model),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            # nn.Tanh(),
            # nn.BatchNorm2d(32),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            # nn.Tanh(),
            # nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(1,10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(d_model),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            # nn.LeakyReLU(negative_slope=0.01),
            # nn.BatchNorm2d(32),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            # nn.LeakyReLU(negative_slope=0.01),
            # nn.BatchNorm2d(32),
        )
        
        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model * 2, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(d_model * 2),
            nn.Conv2d(in_channels=d_model * 2, out_channels=d_model * 2, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(d_model * 2),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model * 2, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(d_model * 2),
            nn.Conv2d(in_channels=d_model * 2, out_channels=d_model * 2, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(d_model * 2),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=d_model, out_channels=d_model * 2, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(d_model * 2),
        )

        self.fc_fusion = nn.Sequential(
            nn.Linear(d_model * 2 * 3, d_model * 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, output_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout),
        )
        
        # # lstm layers
        # self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        # self.fc1 = nn.Linear(64, self.y_len)
    @property
    def output_dim(self) -> int:
        return 64*3
    def forward(self, x):

    
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)  
        
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1) ## (B,C_fusion,T,1)
        
# #         x = torch.transpose(x, 1, 2)
        x = x.permute(0, 2, 1, 3) ## (B,T,C_fusion,1)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2])) ## (B,T,C_fusion*1)
        ## 融合映射到output_dim维度
        x = self.fc_fusion(x)
        return x