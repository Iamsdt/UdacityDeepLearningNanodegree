import torch
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl

# prepare datasets
batch_size = 128

train_data = datasets.MNIST(root='data', train=True,
                            transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=batch_size,
                          shuffle=True, num_workers=0)

# visualize
images, _ = next(iter(train_loader))
images = images.numpy()
img = np.squeeze(images[0])

fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')


# Create Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        # final layer
        self.fc4 = nn.Linear(hidden_size, output_size)
        # drop out
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        # flatten
        x = x.view(28 * 28, -1)
        # pass through fc layer
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        # dropout
        x = self.drop(x)
        return self.fc4(x)


# Create Discriminator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size * 4)
        # final layer
        self.fc4 = nn.Linear(hidden_size * 4, output_size)
        # drop out
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        # pass through fc layer
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        # dropout
        x = self.drop(x)
        return torch.tanh(self.fc4(x))


# hyper parameters
# Discriminator hyperparams
# Size of input image to discriminator (28*28)
input_size = 28 * 28
# Size of discriminator output (real or fake)
d_output_size = 1
# Size of last hidden layer in the discriminator
d_hidden_size = 32

# Generator hyperparams

# Size of latent vector to give to generator
z_size = 100
# Size of discriminator output (generated image)
g_output_size = 28 * 28
# Size of first hidden layer in the generator
g_hidden_size = 32

# create model
D = Discriminator(input_size, d_hidden_size, d_output_size)
G = Generator(z_size, g_hidden_size, g_output_size)

# move to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = D.to(device)
G = G.to(device)

print(D)
print()
print(G)


# loss
def real_loss(out, smooth=False):
    batch = out.size(0)
    labels = torch.ones(batch)
    if smooth:
        labels = labels * 0.9

    # loss function
    criterion = nn.BCEWithLogitsLoss()
    # move to gpu
    labels = labels.to(device)
    # calculate loss
    loss = criterion(out, labels)
    return loss


def fake_loss(out):
    batch = out.size(0)
    labels = torch.zero_(batch)
    # loss function
    criterion = nn.BCEWithLogitsLoss()
    # move to gpu
    labels = labels.to(device)
    # calculate loss
    loss = criterion(out, labels)
    return loss


# optimizer
lr = 0.001
d_optim = optim.Adam(D.parameters(), lr)
g_optim = optim.Adam(G.parameters(), lr)

# train
# training hyperparams
num_epochs = 100
# keep track of loss and generated, "fake" samples
samples = []
losses = []

print_every = 400
sample_size = 16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()

# train network
D.train()
G.train()
for epoch in range(num_epochs):
    for batch, (images, _) in enumerate(train_loader):
        batch_size = images.size(0)
        # rescaling
        images = images * 2 - 1
        #########################
        ## Train Discriminator ##
        #########################
        # Train with real image
        d_optim.zero_grad()
        images = images.to(device)
        out = D(images)
        r_loss = real_loss(out, smooth=True)

        # Train with fake image
        fake_image = np.random.uniform(-1, 1, size=(batch_size, z_size))
        fake_image = torch.from_numpy(fake_image).float()
        # move to gpu
        fake_image = fake_image.to(device)
        f_loss = D(fake_image)

        # combine loss
        loss = r_loss + f_loss
        loss.backward()
        d_optim.step()

        #########################
        #### Train Generator ####
        #########################
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        z = z.to(device)

        # generate fake image
        fake_image = G(z)
        # pass to discriminator
        D_fake = D(fake_image)
        g_loss = real_loss(D_fake)

        # update weight
        g_loss.backward()
        g_optim.step()

        if batch % print_every == 0:
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                epoch + 1, num_epochs, loss.item(), g_loss.item()))

    ## AFTER EACH EPOCH##
    # append discriminator loss and generator loss
    losses.append((loss.item(), g_loss.item()))

    # generate and save sample, fake images
    G.eval()  # eval mode for generating samples
    fixed_z = fixed_z.to(device)
    samples_z = G(fixed_z)
    samples.append(samples_z)
    G.train()  # back to train mode

# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()


# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28, 28)), cmap='Greys_r')


# Load samples from generator, taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)

# -1 indicates final epoch's samples (the last in the list)
view_samples(-1, samples)

rows = 10  # split epochs into 10, so 100/10 = every 10 epochs
cols = 6
fig, axes = plt.subplots(figsize=(7, 12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(samples[::int(len(samples) / rows)], axes):
    for img, ax in zip(sample[::int(len(sample) / cols)], ax_row):
        img = img.detach().cpu()
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

# randomly generated, new latent vectors
sample_size = 16
rand_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
rand_z = torch.from_numpy(rand_z).float()

G.eval()  # eval mode
# generated samples
rand_images = G(rand_z.cuda())

# 0 indicates the first set of samples in the passed in list
# and we only have one batch of samples, here
view_samples(0, [rand_images])
