import torch
from torch import nn, functional as F, optim
from torch.utils.data import DataLoader
from pathlib import Path
from Miniproject_1 import device

### For mini - project 1
class Model(nn.Module):
    def __init__( self ) -> None :
        ## instantiate model + optimizer + loss function + any other stuff you need
        super().__init__()

        self.enc_conv0 = nn.Conv2d(in_channels=3,  out_channels=48, kernel_size=(3, 3), padding='same')
        self.enc_conv1 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding='same')
        self.enc_conv2 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding='same')
        self.enc_conv3 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding='same')
        self.enc_conv4 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding='same')
        self.enc_conv5 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding='same')
        self.enc_conv6 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding='same')
        #concat5
        self.dec_conv5a = nn.Conv2d(in_channels=96, out_channels=96,     kernel_size=(3, 3), padding='same')
        self.dec_conv5b = nn.Conv2d(in_channels=96, out_channels=96,     kernel_size=(3, 3), padding='same')
    
        #concat4
        self.dec_conv4a = nn.Conv2d(in_channels=144, out_channels=96,    kernel_size=(3, 3), padding='same')
        self.dec_conv4b = nn.Conv2d(in_channels=96,  out_channels=96,    kernel_size=(3, 3), padding='same')
    
        #concat3
        self.dec_conv3a = nn.Conv2d(in_channels=144, out_channels=96,    kernel_size=(3, 3), padding='same')
        self.dec_conv3b = nn.Conv2d(in_channels=96,  out_channels=96,    kernel_size=(3, 3), padding='same')
    
        #concat2
        self.dec_conv2a = nn.Conv2d(in_channels=144, out_channels=96,    kernel_size=(3, 3), padding='same')
        self.dec_conv2b = nn.Conv2d(in_channels=96,  out_channels=96,    kernel_size=(3, 3), padding='same')
    
        #concat1
        self.dec_conv1a = nn.Conv2d(in_channels=96 + 3, out_channels=64, kernel_size=(3, 3), padding='same')
        self.dec_conv1b = nn.Conv2d(in_channels=64,     out_channels=32, kernel_size=(3, 3), padding='same')
        self.dec_conv1c = nn.Conv2d(in_channels=32,     out_channels=3,  kernel_size=(3, 3), padding='same')       

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        self.optimizer = None
        self.criterion = nn.MSELoss()

    def load_pretrained_model( self ) -> None :
        ## This loads the parameters saved in bestmodel.pth into the model
        model_path = Path(__file__).parent / "bestmodel.pth"
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)

    def train( self, train_input, train_target, num_epochs ) -> None :
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the
        # same images , which only differs from the input by their noise .
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-08)
        print_every = 100
        best_loss = 2e9

        train_input = train_input.double() / 255.0
        train_target = train_target.double() / 255.0
        train_input = DataLoader(train_input, batch_size=64, shuffle=False)
        train_target = DataLoader(train_target, batch_size=64, shuffle=False)

        print('Training Starts')
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch, (X, Y) in enumerate(zip(train_input, train_target)):
                self.optimizer.zero_grad()

                outputs = self.forward(X)
                loss = self.criterion(outputs, Y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if batch % print_every == print_every - 1:
                    loss = running_loss / print_every
                    print(f'[{epoch + 1}, {batch + 1:5d}] loss: {loss:.5f}')
                    running_loss = 0.0
                    if loss < best_loss:
                        best_loss = loss
                        torch.save(self.state_dict(), 'bestmodel_test.pth')
        print('Finised Training')


    def predict( self, test_input ) -> torch.Tensor :
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained
        # or the loaded network .
        #: returns a tensor of the size (N1 , C, H, W)
        test_input = test_input.double() / 255.0
        with torch.no_grad():
            output = self.forward(test_input)
            output = output * 255.0
            return torch.clip(output, 0.0, 255.0)
    
    def forward(self, x):
        skips = []
        skips.append(x)

        x = self.enc_conv0(x)
        x = self.leaky_relu(x)
        x = self.enc_conv1(x)
        x = self.leaky_relu(x)
        x = self.pool(x)
        skips.append(x)

        x = self.enc_conv2(x)
        x = self.leaky_relu(x)
        x = self.pool(x)
        skips.append(x)

        x = self.enc_conv3(x)
        x = self.leaky_relu(x)
        x = self.pool(x)
        skips.append(x)

        x = self.enc_conv4(x)
        x = self.leaky_relu(x)
        x = self.pool(x)
        skips.append(x)

        x = self.enc_conv5(x)
        x = self.leaky_relu(x)
        x = self.pool(x)
        x = self.enc_conv6(x)
        x = self.leaky_relu(x)

        #-----------------------------------------

        x = self.upsample(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec_conv5a(x)
        x = self.leaky_relu(x)
        x = self.dec_conv5b(x)
        x = self.leaky_relu(x)

        x = self.upsample(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec_conv4a(x)
        x = self.leaky_relu(x)
        x = self.dec_conv4b(x)
        x = self.leaky_relu(x)

        x = self.upsample(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec_conv3a(x)
        x = self.leaky_relu(x)
        x = self.dec_conv3b(x)
        x = self.leaky_relu(x)

        x = self.upsample(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec_conv2a(x)
        x = self.leaky_relu(x)
        x = self.dec_conv2b(x)
        x = self.leaky_relu(x)

        x = self.upsample(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec_conv1a(x)
        x = self.leaky_relu(x)
        x = self.dec_conv1b(x)
        x = self.leaky_relu(x)

        x = self.dec_conv1c(x)

        return x
