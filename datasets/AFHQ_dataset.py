from PIL import Image
from glob import glob
import os
from torch.utils.data import Dataset
import torchvision.transforms as tfs

# copy


class AFHQ_dataset(Dataset):
    def __init__(
        self, image_root, transform=None, mode="train", animal_class="dog", img_size=256, class_name=None
    ):
        super().__init__()
        # self.image_paths = glob(os.path.join(image_root, mode, animal_class, "*.jpg"))
        self.transform = transform
        self.img_size = img_size
        # print('files with ILS')
        # print('afhq dog files', os.listdir("data/afhq/raw_images/train/images"))
        # get only files from imagenet home subset
        print("getting imagenet house images")


        if class_name == None:
            print("WARNING: IMAGENET class name is blank")
            
        files = [
            f"data/afhq/raw_images/train/images/_{class_name}" + f
            for f in os.listdir(
                f"data/afhq/raw_images/train/images/{class_name}"
            )
            if os.path.isfile(
                os.path.join(
                    f"data/afhq/raw_images/train/images/{class_name}", f
                )
            )
            
        ]
        self.image_paths = files

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        print("getting", image_path)
        print("convert PIL to RGB")
        x = Image.open(image_path).convert("RGB")
        x = x.resize((self.img_size, self.img_size))
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.image_paths)


################################################################################


def get_afhq_dataset(data_root, config):
    train_transform = tfs.Compose(
        [tfs.ToTensor(), tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)]
    )

    test_transform = tfs.Compose(
        [tfs.ToTensor(), tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)]
    )

    train_dataset = AFHQ_dataset(
        data_root,
        transform=train_transform,
        mode="train",
        animal_class="dog",
        img_size=config.data.image_size,
    )
    test_dataset = AFHQ_dataset(
        data_root,
        transform=test_transform,
        mode="val",
        animal_class="dog",
        img_size=config.data.image_size,
    )

    return train_dataset, test_dataset
