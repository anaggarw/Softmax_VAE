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
input_dim = 32 * 32 * 3
latent_dim = 900  # number of categories (CIFAR-100 has 100 classes)
output_dim = input_dim
batch_size = 128
num_epochs = 2000
learning_rate = 3e-4

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

# Define the VAE model
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(512, 256)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.prelu1(self.fc1(x))
        x = self.prelu2(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)  # Apply softmax along the latent dimension

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(256, 512)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.prelu1(self.fc1(x))
        x = self.prelu2(self.fc2(x))
        x = self.fc3(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, output_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

def ssim(img1, img2, window_size=11, size_average=True):
    # Ensure input images are in the right format
    if img1.size() != img2.size():
        raise ValueError("Input images must have the same dimensions.")
    
    # Convert images to float32 if they are not already
    img1 = img1.type(torch.FloatTensor)
    img2 = img2.type(torch.FloatTensor)

    # Define the gaussian window
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    # Create the gaussian kernel for SSIM
    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    window = create_window(window_size, img1.size(1))

    # Calculate SSIM
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
    mse_loss = F.mse_loss(reconstructed_x, x, reduction='sum')
    ssim_loss = 1 - ssim(reconstructed_x.view(-1, 3, 32, 32), x.view(-1, 3, 32, 32))

    # Linearly transition from SSIM to MSE
    alpha = epoch / total_epochs  # Linearly increase from 0 to 1
    loss = (1 - alpha) * ssim_loss + alpha * mse_loss

    return loss

# Initialize the model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(input_dim, latent_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with tqdm for progress tracking
total_epochs = num_epochs
for epoch in range(total_epochs):
    epoch_loss = 0
    with tqdm(total=len(trainloader), desc=f'Epoch {epoch+1}/{total_epochs}', unit='batch') as pbar:
        for images, _ in trainloader:
            images = images.view(-1, input_dim).to(device)

            optimizer.zero_grad()
            reconstructed_x, z = model(images)
            loss = custom_loss_function(reconstructed_x, images, epoch, total_epochs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': epoch_loss / (pbar.n + 1)})
            pbar.update()

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

# Conditional probability function
def conditional_probability(z, condition):
    conditional_z = z * condition
    conditional_z /= conditional_z.sum(dim=1, keepdim=True)
    return conditional_z

# Feature importance function
def feature_importance(z, model, original_img):
    importances = []
    for i in range(z.size(1)):
        z_perturbed = z.clone()
        z_perturbed[0, i] = 0
        reconstructed_img = model.decoder(z_perturbed).view(3, 32, 32)
        importance = F.mse_loss(reconstructed_img, original_img).item()
        importances.append(importance)
    return importances

# Load a batch of test images for visualization
dataiter = iter(testloader)
images, labels = next(dataiter)#dataiter.next()
images = images.view(-1, input_dim).to(device)

# Encode and decode a test image for reconstruction
encoded_z = model.encoder(images[0].unsqueeze(0))
decoded_img = model.decoder(encoded_z).view(3, 32, 32).cpu().detach().numpy()

# Display the original and reconstructed image
fig, axes = plt.subplots(1, 2)
axes[0].imshow(images[0].view(3, 32, 32).cpu().numpy().transpose(1, 2, 0))
axes[0].set_title('Original Image')
axes[1].imshow(decoded_img.transpose(1, 2, 0))
axes[1].set_title('Reconstructed Image')
plt.show()

# Interpolate between two test images
z1 = model.encoder(images[0].unsqueeze(0))
z2 = model.encoder(images[1].unsqueeze(0))
interpolated_z = categorical_interpolation(z1, z2, num_steps=10)
interpolated_images = [model.decoder(z).view(3, 32, 32).cpu().detach().numpy().transpose(1, 2, 0) for z in interpolated_z]

# Display the interpolated images
fig, axes = plt.subplots(1, len(interpolated_images), figsize=(20, 2))
for ax, img in zip(axes, interpolated_images):
    ax.imshow(img)
    ax.axis('off')
plt.show()

# Perform Bayesian inference with a uniform prior
prior = torch.ones_like(encoded_z) / encoded_z.size(1)
posterior = bayesian_inference(encoded_z, prior)
decoded_img_posterior = model.decoder(posterior).view(3, 32, 32).cpu().detach().numpy()

# Display the posterior decoded image
plt.imshow(decoded_img_posterior.transpose(1, 2, 0))
plt.axis('off')
plt.title('Posterior Decoded Image')
plt.show()

# Apply a condition to the latent space
condition = torch.tensor([[1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).to(device)  # Boosting the third category
conditional_z = conditional_probability(encoded_z, condition)
decoded_img_conditional = model.decoder(conditional_z).view(3, 32, 32).cpu().detach().numpy()

# Display the conditional decoded image
plt.imshow(decoded_img_conditional.transpose(1, 2, 0))
plt.axis('off')
plt.title('Conditional Decoded Image')
plt.show()

# Visualize the latent space with t-SNE
z_batch = torch.softmax(model.encoder(images), dim=1).cpu().detach().numpy()
labels_batch = labels.cpu().numpy()

# Perform t-SNE
tsne = TSNE(n_components=2)
z_tsne = tsne.fit_transform(z_batch)

# Plot the t-SNE results
plt.figure(figsize=(10, 8))
sns.scatterplot(x=z_tsne[:, 0], y=z_tsne[:, 1], hue=labels_batch, palette='tab20')
plt.title('t-SNE Visualization of Latent Space')
plt.show()

# Compute and display feature importance
importances = feature_importance(encoded_z, model, images[0].view(3, 32, 32).cpu())

plt.bar(range(len(importances)), importances)
plt.xlabel('Latent Feature')
plt.ylabel('Importance (MSE Increase)')
plt.title('Feature Importance for Reconstruction')
plt.show()
