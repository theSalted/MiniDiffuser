import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam

import os
from datetime import datetime

# UTILITIES
def generate_unique_folder_name(base_folder):
    """Function to generate a unique folder name by adding a number if it already exists"""
    counter = 1
    unique_folder = base_folder
    while os.path.exists(unique_folder):
        unique_folder = f"{base_folder}-{counter}"
        counter += 1
    return unique_folder + '/'

def generate_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder, mode=0o777, exist_ok=True)

# MODELS
def get_model():
    block_out_channels=(128, 128, 256, 256, 512, 512)
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    )
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
    )
    return UNet2DModel(block_out_channels=block_out_channels,out_channels=3, in_channels=3, up_block_types=up_block_types, down_block_types=down_block_types, add_attention=True)

@torch.no_grad()
def iadb(model, x0, nb_step):
    x_alpha = x0
    for t in range(nb_step):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)

        d = model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

    return x_alpha

# MAIN
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Cuda is available")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available")

# Generating a base folder name
current_time = datetime.now()
model_name = current_time.strftime('%y%m%d') + '-cifar-fp32'
base_folder = './results/' + model_name

# Generate folders
DATASET_FOLDER = './datasets/cifar10/'
RESULT_FOLDER = generate_unique_folder_name(base_folder)

transform = transforms.Compose([transforms.Resize(32),transforms.CenterCrop(32), transforms.RandomHorizontalFlip(0.5),transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(root=DATASET_FOLDER, train=True,
                                        download=True, transform=transform)
                                        
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

model = get_model()
model = model.to(device)

optimizer = Adam(model.parameters(), lr=1e-4)
nb_iter = 0

print('Start training')
for current_epoch in range(100):
    for i, data in enumerate(dataloader):
        x1 = (data[0].to(device)*2)-1
        x0 = torch.randn_like(x1)
        bs = x0.shape[0]

        alpha = torch.rand(bs, device=device)
        x_alpha = alpha.view(-1,1,1,1) * x1 + (1-alpha).view(-1,1,1,1) * x0
        
        d = model(x_alpha, alpha)['sample']
        loss = torch.sum((d - (x1-x0))**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        nb_iter += 1

        if nb_iter % 200 == 0:
            with torch.no_grad():
                generate_folder(RESULT_FOLDER)
                print(f'Save export {nb_iter} - loss {loss}')
                sample = (iadb(model, x0, nb_step=128) * 0.5) + 0.5
                torchvision.utils.save_image(sample, f'{RESULT_FOLDER}preview_{str(nb_iter).zfill(8)}.png')
                torch.save(model.state_dict(), f'{RESULT_FOLDER}weights.ckpt')
