from torch.utils.data import Dataset
import cv2
import os

class syn_signs(Dataset):
    def __init__(self, root = "./data/syn_signs/Images", transform = False):
        img_paths = []
        labels = []

        for label in os.listdir(root):
            label_dir = os.path.join(root, label)

            for img in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img)
                img_paths.append(img_path)
                labels.append(int(label))

        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

