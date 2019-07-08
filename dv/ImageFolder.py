from torch.utils.data import Dataset, DataLoader
import scipy.io
import numpy as np
import os
from PIL import Image

class CarsDataset(Dataset):
    """
    3D Object Representations for Fine-Grained Categorization
    Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei
    4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013 (3dRR-13). Sydney, Australia. Dec. 8, 2013.
    
    http://ai.stanford.edu/~jkrause/cars/car_dataset.html
    """

    def __init__(self, mat_anno, data_dir, car_names, transform=None):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.full_data_set = scipy.io.loadmat(mat_anno)
        self.car_annotations = self.full_data_set['annotations']
        self.car_annotations = self.car_annotations[0]

        self.car_names = scipy.io.loadmat(car_names)['class_names']
        self.car_names = np.array(self.car_names[0])

        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
        image = Image.open(img_name)
        car_class = self.car_annotations[idx][-2][0][0]
        
        if len(np.array(image).shape) < 3:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, car_class

    def map_class(self, id):
        id = np.ravel(id)
        ret = self.car_names[id - 1][0][0]
        return ret

    def show_batch(self, img_batch, class_batch):

        for i in range(img_batch.shape[0]):
            ax = plt.subplot(1, img_batch.shape[0], i + 1)
            title_str = self.map_class(int(class_batch[i]))
            img = np.transpose(img_batch[i, ...], (1, 2, 0))
            ax.imshow(img)
            ax.set_title(title_str.__str__(), {'fontsize': 5})
            plt.tight_layout()
