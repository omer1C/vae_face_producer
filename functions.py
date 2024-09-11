import torch.nn as nn
import torch
IMAGE_SIZE = 64

class Encoder(nn.Module):

    def __init__(self, in_channels, num_hiddens, latent):
        super(Encoder, self).__init__()

        self.num_hiddens = num_hiddens
        self.latent = latent
        self.in_channels = in_channels
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(num_hiddens),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(num_hiddens, num_hiddens * 2, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(num_hiddens * 2),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(num_hiddens * 2, num_hiddens * 4, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(num_hiddens * 4),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(num_hiddens * 4, num_hiddens * 8, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(num_hiddens * 8),
            nn.ReLU()
        )
        H4, W4, C4 = LinearDimCalc(IMAGE_SIZE,IMAGE_SIZE,self.in_channels,self.num_hiddens)

        self.fc_mu = nn.Linear(H4 * W4 * C4, latent)  # Insert the input size
        self.fc_logvar = nn.Linear(H4 * W4 * C4, latent)  # Insert the input size

    def forward(self, inputs):
        inputs = self.block1(inputs)
        inputs = self.block2(inputs)
        inputs = self.block3(inputs)
        inputs = self.block4(inputs)
        batch_size = inputs.size(0)
        mu = self.fc_mu(inputs.view(batch_size, -1))
        logvar = self.fc_logvar(inputs.view(batch_size, -1))

        return mu, logvar



class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, latent):
        super(Decoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.in_channels = in_channels
        self.H4, self.W4, self.C4 = LinearDimCalc(IMAGE_SIZE, IMAGE_SIZE, self.in_channels, self.num_hiddens)
        self.fc_dec = nn.Linear(latent, self.H4 * self.W4 * self.C4)  # Insert the output size

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(self.C4, num_hiddens // 2, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(num_hiddens // 2),
            nn.LeakyReLU()
        )

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(num_hiddens // 2, num_hiddens // 4, kernel_size=(4, 4), stride=(2, 2),padding=(1, 1)),
            nn.BatchNorm2d(num_hiddens // 4),
            nn.LeakyReLU()
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(num_hiddens // 4, num_hiddens // 8, kernel_size=(4, 4), stride=(2, 2),padding=(1, 1)),
            nn.BatchNorm2d(num_hiddens // 8),
            nn.LeakyReLU()
        )

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(num_hiddens // 8, num_hiddens // 8, kernel_size=(4, 4), stride=(2, 2),padding=(1, 1)),
            nn.BatchNorm2d(num_hiddens // 8),
            nn.LeakyReLU()
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(num_hiddens // 8, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, inputs):

        inputs = self.fc_dec(inputs)
        inputs = inputs.view(-1, self.C4, self.H4, self.W4)
        inputs = self.block1(inputs)
        inputs = self.block2(inputs)
        inputs = self.block3(inputs)
        inputs = self.block4(inputs)
        x_rec = self.block5(inputs)
        return x_rec

class VAE(nn.Module):
    def __init__(self, enc_in_chnl, enc_num_hidden, dec_in_chnl, dec_num_hidden, latent):
        super(VAE, self).__init__()
        self.encode = Encoder(in_channels=enc_in_chnl, num_hiddens=enc_num_hidden,latent=latent)
        self.decode = Decoder(in_channels=dec_in_chnl, num_hiddens=dec_num_hidden,latent=latent)

    # Reparametrization Trick
    def reparametrize(self, mu, logvar):
      std = torch.exp(0.5 * logvar)
      eps = torch.randn_like(std)
      return eps.mul(std).add_(mu)

    # Initialize Weights
    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar


def LinearDimCalc(H_in, W_in, C_in, num_hiddens):
    # Calculate output dimensions after each block
    H_out = (H_in - 3) // 2 + 1
    W_out = (W_in - 3) // 2 + 1
    C1 = num_hiddens

    H_out = (H_out - 3) // 2 + 1
    W_out = (W_out - 3) // 2 + 1
    C2 = num_hiddens * 2

    H_out = (H_out - 3) // 2 + 1
    W_out = (W_out - 3) // 2 + 1
    C3 = num_hiddens * 4

    H_out = (H_out - 3) // 2 + 1
    W_out = (W_out - 3) // 2 + 1
    C4 = num_hiddens * 8

    return H_out, W_out, C4


