import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm

# Hyperparameters
input_channels = 3
input_dim = 32 * 32 * 3
latent_dim = 900  # Updated latent dimension
output_dim = input_dim
batch_size = 128
num_epochs = 2000  # Total number of epochs
learning_rate = 3e-4  # Updated learning rate
meta_epoch = 10  # Save every 10 epochs

# Load CIFAR-100 dataset
transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# Define the VAE models
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.prelu3 = nn.PReLU()
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.softmax(x, dim=1)  # Apply softmax along the latent dimension

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.prelu1 = nn.PReLU()
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.prelu2 = nn.PReLU()
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.prelu3 = nn.PReLU()
        self.deconv3 = nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 4, 4)
        x = self.prelu1(x)
        x = self.prelu2(self.deconv1(x))
        x = self.prelu3(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

# Define the traditional VAE
class TraditionalEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(TraditionalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.prelu3 = nn.PReLU()
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class TraditionalVAE(nn.Module):
    def __init__(self, latent_dim):
        super(TraditionalVAE, self).__init__()
        self.encoder = TraditionalEncoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

def traditional_loss_function(reconstructed_x, x, mu, logvar, epoch, total_epochs):
    BCE = F.mse_loss(reconstructed_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    alpha = epoch / total_epochs
    loss = (1 - alpha) * KLD + alpha * BCE

    return loss

# Define the custom loss function
def ssim(img1, img2, window_size=11, size_average=True):
    if img1.size() != img2.size():
        raise ValueError("Input images must have the same dimensions.")
    
    img1 = img1.type(torch.FloatTensor)
    img2 = img2.type(torch.FloatTensor)

    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    window = create_window(window_size, img1.size(1))
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def custom_loss_function(reconstructed_x, x, epoch, total_epochs):
    mse_loss = F.mse_loss(reconstructed_x, x, reduction='sum').to(device)
    ssim_loss = (1 - ssim(reconstructed_x, x)).to(device)

    alpha = epoch / total_epochs
    loss = (1 - alpha) * ssim_loss + alpha * mse_loss

    return loss

# Save the model and optimizer state dictionaries
def save_checkpoint(model, optimizer, epoch, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filename)
    print(f"Checkpoint saved at epoch {epoch}")

# Load the model and optimizer state dictionaries
def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    return start_epoch

# Initialize the models, optimizers, and load checkpoints if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae_model = VAE(latent_dim).to(device)
vae_optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)
vae_checkpoint = "vae_checkpoint.pth.tar"

traditional_model = TraditionalVAE(latent_dim).to(device)
traditional_optimizer = optim.Adam(traditional_model.parameters(), lr=learning_rate)
traditional_checkpoint = "traditional_checkpoint.pth.tar"

start_epoch_vae = 0
start_epoch_traditional = 0

try:
    start_epoch_vae = load_checkpoint(vae_model, vae_optimizer, vae_checkpoint)
except FileNotFoundError:
    print("No checkpoint found for new VAE. Starting training from scratch.")

try:
    start_epoch_traditional = load_checkpoint(traditional_model, traditional_optimizer, traditional_checkpoint)
except FileNotFoundError:
    print("No checkpoint found for traditional VAE. Starting training from scratch.")

# Training loop with tqdm for progress tracking
total_epochs = num_epochs

for epoch in range(start_epoch_vae, total_epochs):
    epoch_loss_vae = 0
    epoch_loss_traditional = 0
    
    with tqdm(total=len(trainloader), desc=f'Epoch {epoch+1}/{total_epochs}', unit='batch') as pbar:
        for images, _ in trainloader:
            images = images.to(device)

            # Training new VAE model
            vae_optimizer.zero_grad()
            reconstructed_x_vae, z_vae = vae_model(images)
            loss_vae = custom_loss_function(reconstructed_x_vae.to(device), images, epoch, total_epochs)
            loss_vae.backward()
            vae_optimizer.step()

            epoch_loss_vae += loss_vae.item()

            # Training traditional VAE model
            traditional_optimizer.zero_grad()
            reconstructed_x_traditional, mu_traditional, logvar_traditional = traditional_model(images)
            loss_traditional = traditional_loss_function(reconstructed_x_traditional.to(device), images, mu_traditional, logvar_traditional, epoch, total_epochs)
            loss_traditional.backward()
            traditional_optimizer.step()

            epoch_loss_traditional += loss_traditional.item()

            pbar.set_postfix({
                'loss_vae': epoch_loss_vae / (pbar.n + 1),
                'loss_traditional': epoch_loss_traditional / (pbar.n + 1)
            })
            pbar.update()

    # Save checkpoints at the end of each meta-epoch
    if (epoch + 1) % meta_epoch == 0:
        save_checkpoint(vae_model, vae_optimizer, epoch + 1, vae_checkpoint)
        save_checkpoint(traditional_model, traditional_optimizer, epoch + 1, traditional_checkpoint)

print('Training complete.')

# Interpolation function
def categorical_interpolation(z1, z2, num_steps):
    interpolations = []
    for alpha in np.linspace(0, 1, num_steps):
        z_interpolated = alpha * z1 + (1 - alpha) * z2
        z_interpolated = torch.softmax(z_interpolated, dim=1)
        interpolations.append(z_interpolated)
    return interpolations

# Bayesian inference function
def bayesian_inference(z, prior):
    posterior = z * prior
    posterior /= posterior.sum(dim=1, keepdim=True)
    return posterior

# Conditional probability function for New VAE (Softmax Latent Space)
def conditional_probability_softmax(z, condition):
    z = z * condition  # Apply condition
    z /= z.sum(dim=1, keepdim=True)  # Re-normalize to ensure it's still a probability distribution
    return z

# Conditional probability function for Traditional VAE (Gaussian Latent Space)
def conditional_probability_gaussian(z, condition):
    return z + condition  # Shift latent space by the condition

# Feature importance function
def feature_importance(z, model, original_img):
    importances = []
    original_img = original_img.to(device)  # Ensure original_img is on the same device
    for i in range(z.size(1)):
        z_perturbed = z.clone()
        z_perturbed[0, i] = 0
        reconstructed_img = model.decoder(z_perturbed).view(3, 32, 32)
        importance = F.mse_loss(reconstructed_img, original_img).item()
        importances.append(importance)
    return importances


# Load a batch of test images for visualization
dataiter = iter(testloader)
images, labels = next(dataiter)  # Use next() instead of dataiter.next()

images = images.to(device)

# Encode and decode a test image for reconstruction
encoded_z_vae = vae_model.encoder(images[0].unsqueeze(0))
decoded_img_vae = vae_model.decoder(encoded_z_vae).view(3, 32, 32).cpu().detach().numpy()

encoded_mu_traditional, encoded_logvar_traditional = traditional_model.encoder(images[0].unsqueeze(0))
encoded_z_traditional = traditional_model.reparameterize(encoded_mu_traditional, encoded_logvar_traditional)
decoded_img_traditional = traditional_model.decoder(encoded_z_traditional).view(3, 32, 32).cpu().detach().numpy()

# Display the original and reconstructed images
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(images[0].view(3, 32, 32).cpu().numpy().transpose(1, 2, 0))
axes[0].set_title('Original Image')
axes[1].imshow(decoded_img_vae.transpose(1, 2, 0))
axes[1].set_title('New VAE Reconstructed Image')
axes[2].imshow(decoded_img_traditional.transpose(1, 2, 0))
axes[2].set_title('Traditional VAE Reconstructed Image')
plt.show()

# Interpolate between two test images
z1_vae = vae_model.encoder(images[0].unsqueeze(0))
z2_vae = vae_model.encoder(images[1].unsqueeze(0))
interpolated_z_vae = categorical_interpolation(z1_vae, z2_vae, num_steps=10)
interpolated_images_vae = [vae_model.decoder(z).view(3, 32, 32).cpu().detach().numpy().transpose(1, 2, 0) for z in interpolated_z_vae]

z1_traditional = traditional_model.reparameterize(*traditional_model.encoder(images[0].unsqueeze(0)))
z2_traditional = traditional_model.reparameterize(*traditional_model.encoder(images[1].unsqueeze(0)))
interpolated_z_traditional = categorical_interpolation(z1_traditional, z2_traditional, num_steps=10)
interpolated_images_traditional = [traditional_model.decoder(z).view(3, 32, 32).cpu().detach().numpy().transpose(1, 2, 0) for z in interpolated_z_traditional]

# Display the interpolated images for new VAE
fig, axes = plt.subplots(1, len(interpolated_images_vae), figsize=(20, 2))
for ax, img in zip(axes, interpolated_images_vae):
    ax.imshow(img)
    ax.axis('off')
plt.suptitle('Interpolated Images - New VAE')
plt.show()

# Display the interpolated images for traditional VAE
fig, axes = plt.subplots(1, len(interpolated_images_traditional), figsize=(20, 2))
for ax, img in zip(axes, interpolated_images_traditional):
    ax.imshow(img)
    ax.axis('off')
plt.suptitle('Interpolated Images - Traditional VAE')
plt.show()

# Perform Bayesian inference with a uniform prior
prior = torch.ones_like(encoded_z_vae) / encoded_z_vae.size(1)
posterior_vae = bayesian_inference(encoded_z_vae, prior)
decoded_img_posterior_vae = vae_model.decoder(posterior_vae).view(3, 32, 32).cpu().detach().numpy()

# Display the posterior decoded image for new VAE
plt.imshow(decoded_img_posterior_vae.transpose(1, 2, 0))
plt.axis('off')
plt.title('Posterior Decoded Image - New VAE')
plt.show()

# Apply a condition to the latent space
# New VAE
condition_softmax = torch.ones_like(encoded_z_vae) * 1.1  # Example condition: slightly boost all categories
conditional_z_vae = conditional_probability_softmax(encoded_z_vae, condition_softmax)
decoded_img_conditional_vae = vae_model.decoder(conditional_z_vae).view(3, 32, 32).cpu().detach().numpy()

# Traditional VAE
condition_gaussian = torch.randn_like(encoded_z_traditional) * 0.1  # Example condition: random shift
conditional_z_traditional = conditional_probability_gaussian(encoded_z_traditional, condition_gaussian)
decoded_img_conditional_traditional = traditional_model.decoder(conditional_z_traditional).view(3, 32, 32).cpu().detach().numpy()

# Display the conditional decoded image for new VAE
plt.imshow(decoded_img_conditional_vae.transpose(1, 2, 0))
plt.axis('off')
plt.title('Conditional Decoded Image - New VAE')
plt.show()

# Display the conditional decoded image for traditional VAE
plt.imshow(decoded_img_conditional_traditional.transpose(1, 2, 0))
plt.axis('off')
plt.title('Conditional Decoded Image - Traditional VAE')
plt.show()

# Visualize the latent space with t-SNE for new VAE
z_batch_vae = torch.softmax(vae_model.encoder(images), dim=1).cpu().detach().numpy()
labels_batch = labels.cpu().numpy()

# Perform t-SNE for new VAE
tsne_vae = TSNE(n_components=2)
z_tsne_vae = tsne_vae.fit_transform(z_batch_vae)

# Plot the t-SNE results for new VAE
plt.figure(figsize=(10, 8))
sns.scatterplot(x=z_tsne_vae[:, 0], y=z_tsne_vae[:, 1], hue=labels_batch, palette='tab20')
plt.title('t-SNE Visualization of Latent Space - New VAE')
plt.show()

# Compute and display feature importance for new VAE
importances_vae = feature_importance(encoded_z_vae, vae_model, images[0].view(3, 32, 32).cpu())

plt.bar(range(len(importances_vae)), importances_vae)
plt.xlabel('Latent Feature')
plt.ylabel('Importance (MSE Increase)')
plt.title('Feature Importance for Reconstruction - New VAE')
plt.show()

# Perform t-SNE for traditional VAE
z_batch_traditional = torch.softmax(traditional_model.reparameterize(*traditional_model.encoder(images)), dim=1).cpu().detach().numpy()

# Perform t-SNE for traditional VAE
tsne_traditional = TSNE(n_components=2)
z_tsne_traditional = tsne_traditional.fit_transform(z_batch_traditional)

# Plot the t-SNE results for traditional VAE
plt.figure(figsize=(10, 8))
sns.scatterplot(x=z_tsne_traditional[:, 0], y=z_tsne_traditional[:, 1], hue=labels_batch, palette='tab20')
plt.title('t-SNE Visualization of Latent Space - Traditional VAE')
plt.show()

# Compute and display feature importance for traditional VAE
importances_traditional = feature_importance(encoded_z_traditional, traditional_model, images[0].view(3, 32, 32).cpu())

plt.bar(range(len(importances_traditional)), importances_traditional)
plt.xlabel('Latent Feature')
plt.ylabel('Importance (MSE Increase)')
plt.title('Feature Importance for Reconstruction - Traditional VAE')
plt.show()
