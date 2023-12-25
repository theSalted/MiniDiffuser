# IMPORTS
# ML related
import torch
from torch.utils.data import dataset
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam

# Utilities
import os
import sys
import time
import argparse
from datetime import datetime
from tqdm import tqdm
import asciichartpy

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
    
def convert_floats_to_strings(float_list):
    return [str(f) for f in float_list]
    
def plot_data(data, title):
    terminal_width = os.get_terminal_size().columns
    target_length = max(50, terminal_width - 20)
    downsampled_data = downsample_list(data, target_length)
    plot = asciichartpy.plot(downsampled_data, {'height': 10, 'padding': '      ', 'offset': 5})
    tqdm.write(f'{bcolors.HEADER}{title}{bcolors.ENDC}')
    tqdm.write(plot)
    
def downsample_list(data, target_length):
    if len(data) <= target_length:
        return data
    
    segment_length = len(data) // target_length
    downsampled = []
    
    for i in range(0, len(data), segment_length):
        segment = data[i:i + segment_length]
        representative_value = sum(segment) / len(segment)
        downsampled.append(representative_value)
    
    return downsampled
    
# MODELS
@torch.no_grad()
def iadb(model, x0, nb_step):
    x_alpha = x0
    for t in range(nb_step):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)

        d = model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

    return x_alpha

def find_find_torch_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        tqdm.write(f'{bcolors.OKGREEN}Cuda is available{bcolors.ENDC}')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        tqdm.write(f'{bcolors.OKGREEN}MPS is available{bcolors.ENDC}')
    return device

def get_dataset(root, dataset_name, transform):
    if dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        return train_dataset
    elif dataset_name == "celeba":
        train_dataset = torchvision.datasets.CelebA(root=root, split='train',
        download=True, transform=transform)
        return train_dataset
        
    return None
        
def get_dataloader(dataset_name, dataset_folder, transform, batch_size):
    train_dataset = get_dataset(root = dataset_folder, dataset_name = dataset_name, transform = transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return dataloader
    
class MiniD:
    device = None
    dataset_name = "cifar10"
    DATASET_FOLDER = ""
    RESULT_FOLDER = ""
    dataloader = None
    model = None
    optimizer = None
    save_iter = 200
    x0 = None
    losses = []
    
    def __init__(self, device, dataset_name="cifar10", batch_size=32, res=32, half=False, save_iter=200):
        self.device = device
        self.dataset_name = dataset_name
        # Generating a base folder name
        current_time = datetime.now()
        model_name = current_time.strftime('%y%m%d') + f'-{self.dataset_name}-fp32'
        base_folder = './results/' + model_name
        
        # Generate folders
        self.DATASET_FOLDER = f'./datasets/{self.dataset_name}/'
        self.RESULT_FOLDER = generate_unique_folder_name(base_folder)
        
        # Compose images
        transform = transforms.Compose([transforms.Resize(res),transforms.CenterCrop(res), transforms.RandomHorizontalFlip(0.5),transforms.ToTensor()])
        # Load datasets
       
        self.dataloader = get_dataloader(f'{self.dataset_name}', self.DATASET_FOLDER, transform, batch_size)
        
        m = get_model()
        self.model = m.to(self.device)
        
        if half:
            
            self.model = self.model.half()
            self.optimizer = Adam(self.model.parameters(), lr=1e-4, eps=1e-4)
            tqdm.write(f'{bcolors.OKGREEN}Training at half i.e. fp16 memory format{bcolors.ENDC}')
        else:
            self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        
        
    def start(self):
        nb_iter = 0
        tqdm.write(f'{bcolors.BOLD}Start training on {self.dataset_name}{bcolors.ENDC}')
        for current_epoch in tqdm(range(100), desc='Epoch', unit="epoch", colour="green"):
            for i, data in enumerate(tqdm(self.dataloader, desc=f'Iter (Epoch {current_epoch})', unit="iter", colour="blue")):
                x1 = (data[0].to(self.device)*2)-1
                self.x0 = torch.randn_like(x1)
                bs = self.x0.shape[0]
        
                alpha = torch.rand(bs, device=self.device)
                x_alpha = alpha.view(-1,1,1,1) * x1 + (1-alpha).view(-1,1,1,1) * self.x0
                
                d = self.model(x_alpha, alpha)['sample']
                
                l = torch.sum((d - (x1-self.x0))**2)
        
                self.optimizer.zero_grad()
                l.backward()
                
                self.losses.append(float(f'{l}'))
                
                self.optimizer.step()
                nb_iter += 1
                if nb_iter % 200 == 0:
                    with torch.no_grad():
                        self.save(nb_iter, l=l)
    def save(self, name, l=None):
        generate_folder(self.RESULT_FOLDER)
        message = f'{bcolors.OKCYAN}Saving weights and preview...{bcolors.ENDC}'
        if l != None:
            message = f'{bcolors.OKCYAN}Save weights and preview #{name} (loss {l}){bcolors.ENDC}'
        tqdm.write(message)
        sample = (iadb(self.model, self.x0, nb_step=128) * 0.5) + 0.5
        torchvision.utils.save_image(sample, f'{self.RESULT_FOLDER}preview_{str(name).zfill(8)}.png')
        torch.save(self.model.state_dict(), f'{self.RESULT_FOLDER}weights.ckpt')
        
    def save_losses(self):
        tqdm.write(f'{bcolors.OKCYAN}Saving losses record...{bcolors.ENDC}')
        with open(f'{self.RESULT_FOLDER}losses.txt','w') as tfile:
            tfile.write('\n'.join(convert_floats_to_strings(self.losses)))
        plot_data(self.losses, "Losses Plot - x-axis may be scaled")

# MAIN
# args
parser = argparse.ArgumentParser(description='MINI DIFFUSER - A light diffusion model based on IADB')

parser.add_argument('--b', type=int,
                    help='batch size of dataset')

parser.add_argument('--d', type=str,
                    help='dataset to train on')
                    
parser.add_argument('--fp16',default=False, action=argparse.BooleanOptionalAction, help='set memory format to to half i.e. fp16, and reduce optimizer eps to 1e-4')

parser.add_argument('--r', type=int,
                    help='resolution resize to')
                    
args = parser.parse_args()

print(f'{bcolors.HEADER}MINI DIFFUSER{bcolors.ENDC}')
device = find_find_torch_device()
dataset_name = args.d or "cifar10"
batch_size = args.b or 32
res = args.r or 32
half = args.fp16 or False
minid_model = MiniD(device=device, dataset_name=dataset_name, batch_size=batch_size, res=res, half=half)

try:
    minid_model.start()
except KeyboardInterrupt:
    # crt+c to save and exit
    tqdm.write(f'{bcolors.WARNING}Model Interrupted{bcolors.ENDC}')
    minid_model.save("final")
    minid_model.save_losses()
    sys.exit()
