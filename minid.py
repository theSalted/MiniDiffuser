# IMPORTS
# ML related
import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam

# Utilities
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm

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

# Class for print colors
class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

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

class MiniD:
    
tqdm.write(f'{bcolors.HEADER}Mini Diffuser{bcolors.ENDC}')
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    tqdm.write(f'{bcolors.OKGREEN}Cuda is available{bcolors.ENDC}')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    tqdm.write(f'{bcolors.OKGREEN}MPS is available{bcolors.ENDC}')

# Generating a base folder name
current_time = datetime.now()
model_name = current_time.strftime('%y%m%d') + '-cifar-fp32'
base_folder = './results/' + model_name

# Generate folders
DATASET_FOLDER = './datasets/cifar10/'
RESULT_FOLDER = generate_unique_folder_name(base_folder)

# Compose images
transform = transforms.Compose([transforms.Resize(32),transforms.CenterCrop(32), transforms.RandomHorizontalFlip(0.5),transforms.ToTensor()])

# Load datasets
train_dataset = torchvision.datasets.CIFAR10(root=DATASET_FOLDER, train=True,
                                        download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

model = get_model()
model = model.to(device)

loss = tqdm(desc="Loss", bar_format="{desc}: {n_fmt}", position=2)

optimizer = Adam(model.parameters(), lr=1e-4)
    
# MAIN
def main():
    nb_iter = 0
    tqdm.write(f'{bcolors.BOLD}Start training{bcolors.ENDC}')
    for current_epoch in tqdm(range(100), desc='Epoch', unit="epoch", colour="green"):
        for i, data in enumerate(tqdm(dataloader, desc=f'Iter (Epoch {current_epoch+1})', unit="iter", colour="blue")):
            x1 = (data[0].to(device)*2)-1
            x0 = torch.randn_like(x1)
            bs = x0.shape[0]
    
            alpha = torch.rand(bs, device=device)
            x_alpha = alpha.view(-1,1,1,1) * x1 + (1-alpha).view(-1,1,1,1) * x0
            
            d = model(x_alpha, alpha)['sample']
            
            l = torch.sum((d - (x1-x0))**2)
    
            optimizer.zero_grad()
            l.backward()
            
            # update loss
            loss.update(int(float(f'{l}')))
            
            optimizer.step()
            nb_iter += 1
    
            if nb_iter % 200 == 0:
                with torch.no_grad():
                    generate_folder(RESULT_FOLDER)
                    tqdm.write(f'{bcolors.OKCYAN}Save weights and preview #{nb_iter} (loss {loss}){bcolors.ENDC}')
                    sample = (iadb(model, x0, nb_step=128) * 0.5) + 0.5
                    torchvision.utils.save_image(sample, f'{RESULT_FOLDER}preview_{str(nb_iter).zfill(8)}.png')
                    torch.save(model.state_dict(), f'{RESULT_FOLDER}weights.ckpt')

try:
    main()
except KeyboardInterrupt:
    loss.close()
    tqdm.write(f'{bcolors.WARNING}Model Interrupted (not saved){bcolors.ENDC}')
    sys.exit()
