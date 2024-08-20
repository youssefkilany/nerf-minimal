from torch.utils.data import Dataset, DataLoader
from .load_dataset import load_dataset_data


class NerfDataset(Dataset):
    def __init__(self, imgs, poses, h, w, focal):
        self.imgs = imgs
        self.poses = poses
        self.h = h
        self.w = w
        self.focal = focal

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        return self.imgs[index], self.poses[index], h, w, focal


dataloader_batch_size = 1
dataset_base_path = "F:/nerf/dex_nerf_real_dishwasher"

[h, w, focal], imgs, poses, idx_splits = load_dataset_data(
    dataset_base_path, testskip=10
)

train_dataset_imgs, train_dataset_poses = imgs[idx_splits[0]], poses[idx_splits[0]]
val_dataset_imgs, val_dataset_poses = imgs[idx_splits[1]], poses[idx_splits[1]]
test_dataset_imgs, test_dataset_poses = imgs[idx_splits[2]], poses[idx_splits[2]]

train_dataset = NerfDataset(train_dataset_imgs, train_dataset_poses, h, w, focal)
val_dataset = NerfDataset(val_dataset_imgs, val_dataset_poses, h, w, focal)
test_dataset = NerfDataset(test_dataset_imgs, test_dataset_poses, h, w, focal)

train_dataloader = DataLoader(
    train_dataset, batch_size=dataloader_batch_size, shuffle=True
)
train_dataloader.dataset_settings_hwf = [h, w, focal]

val_dataloader = DataLoader(val_dataset, batch_size=dataloader_batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=dataloader_batch_size)
