import torch
import torch.nn.functional as F

n_features = 16

class LinearVAE(torch.nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()

        # encoder
        self.enc1 = torch.nn.Linear(in_features = 784, out_features = 512)
        self.enc2 = torch.nn.Linear(in_features = 512, out_features = n_features*2)

        # decoder 
        self.dec1 = torch.nn.Linear(in_features = n_features, out_features = 512)
        self.dec2 = torch.nn.Linear(in_features=512, out_features = 784)

    def reparametrize(self, mu, log_var):
        """[summary]

        Args:
            mu ([type]): Mean of the encoders latent space distribution.
            log_var ([type]): log variance of the encoder's latent space.
        """

        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + eps*std # reparametrization.
        return sample

    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, n_features)

        # get mu and log_var.
        mu = x[:, 0, :]
        log_var = x[:, 1, :]

        # get the latent vector through reparametrization.
        z = self.reparametrize(mu, log_var)

        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction, mu, log_var