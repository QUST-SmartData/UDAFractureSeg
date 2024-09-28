import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image


class CoreDataSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(512, 512), mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = mean
        self.files = []

        img_ids = [i_id.strip() for i_id in open(list_path)]
        for name in img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)

        # # crop
        # height, width = image.size
        # top = np.random.randint(0, height - self.crop_size[0])
        # left = np.random.randint(0, width - self.crop_size[1])
        # bottom = top + self.crop_size[0]
        # right = left + self.crop_size[1]
        # image = image.crop((left, top, right, bottom))

        image = np.asarray(image, np.float32)

        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), name


if __name__ == '__main__':
    dst = CoreDataSet("./core/data", 'core_train_list.txt')
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            break
