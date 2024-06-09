from scipy import io
import torch
from torch.utils.data import DataLoader, Dataset
import cv2 as cv


pa100k_label = io.loadmat(r"D:\reid\PA100K\annotation\annotation.mat")
pa100k_data_root = r"D:\reid\PA100K\data\release_data\release_data/"
attr = pa100k_label['attributes']

# print(len(path_pa100k_data_lst))
# print(path_pa100k_label)

width = 32
height = 96


class PA100k(Dataset):
    def __init__(self, mode='train', device=torch.device("cpu")):
        super(PA100k, self).__init__()
        self.x_root = pa100k_data_root
        self.y = torch.tensor(pa100k_label['{}_label'.format(mode)], dtype=torch.float, device=device)
        self.x_lst = pa100k_label['{}_images_name'.format(mode)]
        self.device = device

    def __getitem__(self, idx):
        pth = self.x_root + str(self.x_lst[idx][0][0])
        image = torch.zeros(3, height, width, dtype=torch.float, device=self.device)
        img = cv.imread(pth)
        rate = min([height / img.shape[0], width / img.shape[1]])
        img = cv.resize(img, [int(img.shape[1] * rate), int(img.shape[0] * rate)])
        img = img.transpose([2, 0, 1])[[2, 1, 0], :, :]

        image[:, :img.shape[1], :img.shape[2]] = torch.tensor(img, dtype=torch.float, device=self.device) / 255 * 2 - 1

        return image, self.y[idx]

    def __len__(self):
        return self.y.shape[0]


def loader(mode, device, batch_size):
    return DataLoader(PA100k(mode, device), batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    for b in PA100k():
        x, y = b
        print(x.shape, y.shape)

