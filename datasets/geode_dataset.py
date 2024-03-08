import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pickle
from glob import glob
import os
from torch.utils.data import Dataset
import torchvision.transforms as tfs
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torchvision.transforms as T
import numpy as np


class GEODE_dataset(Dataset):
    def __init__(self, image_root, transform=None, mode="train", img_size=512, region=None, class_name=None):
        super().__init__()
        # self.image_paths = glob(
        #     os.path.join(
        #         image_root,
        #         "raw_images",
        #         mode,
        #         "images",
        #         "pixabay_",
        #         animal_class,
        #         "*.jpg",
        #     )
        # )
        if region == None or class_name == None:
            print("WARNING: GEODE region or class name is blank")

        dir_path = f"./data/geode/{region}_{class_name}/"
        print("in GEODE dataset")
        # print(os.listdir(dir_path))
        files = [
            dir_path + f
            for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f))
        ]
        self.image_paths = files

        print("GeodE house data:", self.image_paths)
        self.transform = transform
        self.img_size = img_size

    def __getitem__(self, index):
        print("getting GeoDE image", self.image_paths[index])
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        x = x.resize((self.img_size, self.img_size))
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.image_paths)


def get_geode_dataset(data_root, config, class_name=None, region=None):
    train_transform = tfs.Compose(
        [tfs.ToTensor(), tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)]
    )

    test_transform = tfs.Compose(
        [tfs.ToTensor(), tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)]
    )

    train_dataset = GEODE_dataset(
        data_root,
        transform=train_transform,
        mode="train",
        img_size=config.data.image_size,
        class_name=class_name,
        region=region
    )
    test_dataset = GEODE_dataset(
        data_root,
        transform=test_transform,
        mode="val",
        img_size=config.data.image_size,
        class_name=class_name,
        region=region,
    )

    return train_dataset, test_dataset


class ImageDatasetMultiLabel(Dataset):
    def __init__(self, img_paths, img_labels, transform=T.ToTensor(), n_col=40):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform
        self.n_col = n_col

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        ID = self.img_paths[index]
        img = Image.open(ID).convert("RGB")
        X = self.transform(img)
        y = np.zeros(self.n_col)
        y[self.img_labels[ID]] = 1

        return X, y, "_"


class ImageDataset(Dataset):
    def __init__(self, img_paths, img_labels, img_regions, transform=T.ToTensor()):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.img_regions = img_regions
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        ID = self.img_paths[index]
        img = Image.open(ID).convert("RGB")
        X = self.transform(img)
        y = self.img_labels[index]
        r = self.img_regions[index]

        return X, y, r


class ImageDataset_imagenet(Dataset):
    def __init__(self, img_paths, img_labels, transform=T.ToTensor()):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        ID = self.img_paths[index]
        img = Image.open(ID).convert("RGB")
        X = self.transform(img)
        y = self.img_labels[index]

        return X, y


def prep_imagenet():
    master_csv = pd.read_csv("../imagenet38_filenames.csv")

    image_names = []  # stores directory path to each image
    obj = []

    all_obj_names = sorted(
        list(master_csv["script_name"].unique())
    )  # gets the 38 categories in alpha order
    print(all_obj_names)

    for idx in master_csv.index:
        fname = master_csv["file_name"][idx]  # country_category_#.jpg
        oname = master_csv["script_name"][idx]  # object category of the image
        image_names.append(".../{}/{}".format(oname, fname))
        obj.append(
            all_obj_names.index(oname)
        )  # number label corresponding to the object classification

        if idx % 1000 == 0:
            print(idx)

    train_names, valtest_names, train_obj, valtest_obj = train_test_split(
        image_names, obj, random_state=42, test_size=0.4
    )
    val_names, test_names, val_obj, test_obj = train_test_split(
        valtest_names, valtest_obj, random_state=42, test_size=0.5
    )

    if not os.path.exists("data"):
        os.mkdir("data")
    with open("data/imagenet_prep.pkl", "wb+") as handle:
        pickle.dump(
            {
                "train": [train_names, train_obj],
                "val": [val_names, val_obj],
                "test": [test_names, test_obj],
            },
            handle,
        )


def prep_geode_38():

    master_csv = pd.read_csv("../geode38_meta.csv")

    image_names = []
    obj = []

    country_to_region = get_country_to_region()  # maps country:number
    region_to_number = get_reg_to_number()  # maps region name:region number
    number_to_region = {
        v: k for (k, v) in region_to_number.items()
    }  # changes map to number:region

    all_obj_names = sorted(
        list(master_csv["script_name"].unique())
    )  # gets the 38 categories in alpha order
    print(all_obj_names)

    for idx in master_csv.index:
        fname = master_csv["file_name"][idx]  # country_category_#.jpg
        oname = master_csv["script_name"][idx]  # object category of the image
        cname = master_csv["ip_country"][idx].replace(" ", "_")  # country of the image
        region = number_to_region[country_to_region[cname]]
        if region == "Africa":  # change for region
            image_names.append(
                ".../{}/{}/{}".format(
                    number_to_region[country_to_region[cname]], cname, fname
                )
            )
            obj.append(
                all_obj_names.index(oname)
            )  # number label corresponding to the object classification

            if idx % 1000 == 0:
                print(idx)

    train_names, valtest_names, train_obj, valtest_obj = train_test_split(
        image_names, obj, random_state=42, test_size=0.4
    )
    val_names, test_names, val_obj, test_obj = train_test_split(
        valtest_names, valtest_obj, random_state=42, test_size=0.5
    )

    if not os.path.exists("data"):
        os.mkdir("data")
    with open("data/geode_prep_africa.pkl", "wb+") as handle:  # change for each region
        pickle.dump(
            {
                "train": [train_names, train_obj],
                "val": [val_names, val_obj],
                "test": [test_names, test_obj],
            },
            handle,
        )


def prep_geode():

    # master_csv = pd.read_csv('../geode_paths.csv')
    master_csv = pd.read_csv("../geode/index.csv")

    image_names = []
    obj = []
    reg = []

    country_to_region = get_country_to_region()  # maps country:number
    region_to_number = get_reg_to_number()  # maps region name:region number
    number_to_region = {
        v: k for (k, v) in region_to_number.items()
    }  # changes map to number: region

    all_obj_names = sorted(
        list(master_csv["script_name"].unique())
    )  # gets the 38 categories in alpha order
    print(all_obj_names)

    for idx in master_csv.index:
        fname = master_csv["file_name"][idx]  # country_category_#.jpg
        oname = master_csv["script_name"][idx]  # object category of the image
        cname = master_csv["ip_country"][idx].replace(" ", "_")  # country of the image
        image_names.append(
            ".../{}/{}/{}".format(
                number_to_region[country_to_region[cname]], cname, fname
            )
        )
        obj.append(
            all_obj_names.index(oname)
        )  # number label corresponding to the object classification
        reg.append(country_to_region[cname])  # appends region number of image

        if idx % 1000 == 0:
            print(idx)

    train_names, valtest_names, train_obj, valtest_obj, train_reg, valtest_reg = (
        train_test_split(image_names, obj, reg, random_state=42, test_size=0.4)
    )
    val_names, test_names, val_obj, test_obj, val_reg, test_reg = train_test_split(
        valtest_names, valtest_obj, valtest_reg, random_state=42, test_size=0.5
    )

    if not os.path.exists("data"):
        os.mkdir("data")
    with open("data/geode_prep.pkl", "wb+") as handle:
        pickle.dump(
            {
                "train": [train_names, train_obj, train_reg],
                "val": [val_names, val_obj, val_reg],
                "test": [test_names, test_obj, test_reg],
            },
            handle,
        )


def prep_imagenet_excluded():
    imagenet_dset_full = pickle.load(open(".../imagenet_prep.pkl", "rb"))
    imagenet_dset = pickle.load(open(".../imagenet_33383indices.pkl", "rb"))
    imagenet_random_sample = np.array(imagenet_dset["index"][0])
    print("Imagenet Training Size: " + str(len(imagenet_random_sample)))
    all_indices = np.arange(len(imagenet_dset_full["train"][0]))
    print("Full Imagenet Training Size: " + str(len(all_indices)))
    excluded_pics = np.array(
        [i for i in all_indices if i not in imagenet_random_sample]
    )
    print("Excluded Pics Size: " + str(len(excluded_pics)))

    full_38_cat = []
    for obj in range(38):
        category_pics = []
        for pic_index in excluded_pics:
            if imagenet_dset_full["train"][1][pic_index] == obj:
                category_pics.append(pic_index)
        full_38_cat.append(category_pics)

    if not os.path.exists(".../data"):
        os.mkdir(".../data")
    with open(".../data/imagenet33383_excludedIndices.pkl", "wb+") as handle:
        pickle.dump(
            {
                "0": [full_38_cat[0]],
                "1": [full_38_cat[1]],
                "2": [full_38_cat[2]],
                "3": [full_38_cat[3]],
                "4": [full_38_cat[4]],
                "5": [full_38_cat[5]],
                "6": [full_38_cat[6]],
                "7": [full_38_cat[7]],
                "8": [full_38_cat[8]],
                "9": [full_38_cat[9]],
                "10": [full_38_cat[10]],
                "11": [full_38_cat[11]],
                "12": [full_38_cat[12]],
                "13": [full_38_cat[13]],
                "14": [full_38_cat[14]],
                "15": [full_38_cat[15]],
                "16": [full_38_cat[16]],
                "17": [full_38_cat[17]],
                "18": [full_38_cat[18]],
                "19": [full_38_cat[19]],
                "20": [full_38_cat[20]],
                "21": [full_38_cat[21]],
                "22": [full_38_cat[22]],
                "23": [full_38_cat[23]],
                "24": [full_38_cat[24]],
                "25": [full_38_cat[25]],
                "26": [full_38_cat[26]],
                "27": [full_38_cat[27]],
                "28": [full_38_cat[28]],
                "29": [full_38_cat[29]],
                "30": [full_38_cat[30]],
                "31": [full_38_cat[31]],
                "32": [full_38_cat[32]],
                "33": [full_38_cat[33]],
                "34": [full_38_cat[34]],
                "35": [full_38_cat[35]],
                "36": [full_38_cat[36]],
                "37": [full_38_cat[37]],
            },
            handle,
        )


def prep_imagenet_subsec():
    imagenet_dset = pickle.load(open(".../imagenet_prep.pkl", "rb"))
    imagenet_num_images_ratio = 33400 / len(imagenet_dset["train"][0])
    imagenet_random_sample = []

    for obj in range(38):
        sum = 0
        subset = []
        for idx in range(len(imagenet_dset["train"][1])):
            if obj == imagenet_dset["train"][1][idx]:
                sum = sum + 1
                subset.append(idx)
        num_images = np.floor(imagenet_num_images_ratio * sum).astype(int)
        subset_sample = np.random.choice(subset, num_images, replace=False)
        imagenet_random_sample = [*imagenet_random_sample, *subset_sample]

    if not os.path.exists(".../data"):
        os.mkdir(".../data")
    with open(".../data/imagenet_33383indices.pkl", "wb+") as handle:
        pickle.dump({"index": [imagenet_random_sample]}, handle)


from sklearn.linear_model import LogisticRegression
import torch
import numpy as np
from sklearn.metrics import average_precision_score


def get_country_to_region():
    return {
        "Angola": 0,
        "Botswana": 0,
        "Cameroon": 0,
        "Egypt": 0,
        "Ghana": 0,
        "Nigeria": 0,
        "South_Africa": 0,
        "Argentina": 1,
        "Brazil": 1,
        "Colombia": 1,
        "Mexico": 1,
        "Uruguay": 1,
        "Venezuela": 1,
        "China": 2,
        "Hong_Kong": 2,
        "Japan": 2,
        "South_Korea": 2,
        "Taiwan": 2,
        "Bulgaria": 3,
        "France": 3,
        "Greece": 3,
        "Italy": 3,
        "Poland": 3,
        "Romania": 3,
        "Switzerland": 3,
        "United_Kingdom": 3,
        "Cyprus": 3,
        "Germany": 3,
        "Ireland": 3,
        "Netherlands": 3,
        "Portugal": 3,
        "Spain": 3,
        "Ukraine": 3,
        "Indonesia": 4,
        "Malaysia": 4,
        "Philippines": 4,
        "Singapore": 4,
        "Thailand": 4,
        "Jordan": 5,
        "Saudi_Arabia": 5,
        "Turkey": 5,
        "United_Arab_Emirates": 5,
        "Yemen": 5,
    }


def get_reg_to_number():
    return {
        "Africa": 0,
        "Americas": 1,
        "EastAsia": 2,
        "Europe": 3,
        "SouthEastAsia": 4,
        "WestAsia": 5,
    }


def train_linear_with_reg(Xtrain, ytrain, Xval, yval, regs=[10, 1, 0.1, 0.01, 0.001]):

    best_clf = None
    best_acc = 0

    for c in regs:
        clf = LogisticRegression(C=c)
        clf.fit(Xtrain, ytrain)

        acc = clf.score(Xval, yval)

        if acc > best_acc:
            best_acc = acc
            best_clf = clf

    print(best_acc)
    return best_clf, best_acc


def train_pytorch_linear(Xtrain, ytrain, Xval, yval, curr_attr):

    model = torch.nn.Linear(2048, 40)

    Xtrain = torch.Tensor(Xtrain)
    ytrain = torch.Tensor(ytrain).to(dtype=torch.long)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    batch = 512
    N = len(Xtrain) // batch + 1

    model.train()
    model.cuda()
    best_model = None
    best_acc = 0

    for e in range(200):
        model.train()
        model.cuda()

        for t in range(N):
            Xbatch = Xtrain[t * batch : (t + 1) * batch].to(torch.device("cuda"))
            ybatch = ytrain[t * batch : (t + 1) * batch].to(torch.device("cuda"))

            scores = model(Xbatch)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer.zero_grad()
            loss = criterion(scores, ybatch)
            loss.backward()

            optimizer.step()

        model.eval()
        scores = (
            model(torch.Tensor(Xval).to(torch.device("cuda"))).detach().cpu().numpy()
        )

        acc = average_precision_score(
            np.where(np.array(yval) == curr_attr, 1, 0).squeeze(),
            scores[:, curr_attr].squeeze(),
        )

        if acc > best_acc:
            best_acc = acc
            best_model = model.cpu().state_dict()

    return best_model, best_acc


if __name__ == "__main__":
    # prep_geode_38()
    prep_geode()
