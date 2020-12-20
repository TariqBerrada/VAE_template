import torch
import torchvision
import argparse
import matplotlib
import matplotlib.pyplot as plt
import model

import tqdm

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms

matplotlib.pyplot.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=30, type = int, help = 'number of epochs to train the VAE for.')
args = vars(parser.parse_args())

epochs = args['epochs']
batch_size = 64
lr = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defining the transforms for the MNIST dataset.
transform = transforms.Compose([transforms.ToTensor(),])
train_data = datasets.FashionMNIST(root = '../input/data', train = True, download = True, transform = transform)

val_data = datasets.FashionMNIST(root = '../input_data', train = False, download = True, transform = transform)

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False)

# Initializing model, optimizer and loss function.
model = model.LinearVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.BCELoss(reduction = 'sum')

def final_loss(bce_loss, mu, logvar):
    """Return loss function composed of a KL-Divergence term and a BCE term.
        KL-div = .5*sum(1+log(sigma^2) - mu^2 - sigma^2)

    Args:
        bce_loss ([type]): [description]
        mu ([type]): [description]
        logvar ([type]): [description]
    """
    BCE = bce_loss
    KLD = -.5*torch.sum(1+logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    for i, data in tqdm.tqdm(enumerate(dataloader), total = int(len(train_data)/dataloader.batch_size)):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        
        reconstruction, mu, logvar = model(data)
        
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        
        running_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss

def validate(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(dataloader), total = int(len(val_data)/dataloader.batch_size)):
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()

            # Save last batch of every epoch.
            if i == int(len(val_data)/dataloader.batch_size) - 1:
                num_rows = 8
                both = torch.cat((data.view(batch_size, 1, 28, 28)[:8], reconstruction.view(batch_size, 1, 28, 28)[:8]))
                save_image(both.cpu(), './outputs/output_%d.png'%epoch, nrow = num_rows)
    
    val_loss = running_loss/len(dataloader.dataset)
    return val_loss

train_loss = []
val_loss = []

for epoch in tqdm.tqdm(range(epochs), desc = 'train_epoch'):
    train_epoch_loss = fit(model, train_loader)
    val_epoch_loss = validate(model, val_loader, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(' - train_loss : ', train_epoch_loss)
    print(' - val_loss : ', val_epoch_loss)

print(train_loss)
print(val_loss)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(train_loss, label = 'train')
ax.plot(val_loss, label = 'validate')
plt.legend()
plt.show()