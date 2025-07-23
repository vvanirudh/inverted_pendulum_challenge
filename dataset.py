import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class PendulumDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory containing the expert trajectories
            transform: Optional transforms to apply to the images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # Load all trajectory files
        trajectory_files = [f for f in os.listdir(data_dir) if f.startswith('trajectory_') and f.endswith('.npy')]

        for traj_file in trajectory_files:
            # Load the trajectory data (images and control inputs)
            traj_path = os.path.join(data_dir, traj_file)
            traj_data = np.load(traj_path, allow_pickle=True)

            # Each trajectory contains 50 (image, control) pairs
            for i in range(len(traj_data)):
                img_path = traj_data[i]['image_path']
                control = traj_data[i]['control']
                self.samples.append((img_path, control))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, control = self.samples[idx]

        # Load and convert image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms if available
        if self.transform:
            image = self.transform(image)

        # Convert control to tensor
        control = torch.tensor(control, dtype=torch.float32)

        return image, control
