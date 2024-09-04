import json
import cv2
import numpy as np

from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms


class BDD100K(Dataset):
    def __init__(self, mode="train"):
        print("Loading BDD100K dataset...")

        self.img_dir = f"/workspace/BDD_processed/{mode}/bdd100k/images/100k/{mode}"
        self.canny_dir = f"/workspace/BDD_processed/{mode}/bdd100k/canny/100k/{mode}"
        f = open(f"/workspace/BDD_processed/labels/det_20/det_{mode}.json")
        data = json.load(f)
        self.img_fnames = []
        self.canny_fnames = []
        self.time_of_day = []
        for image in data:
            if mode == "val" and image["attributes"]["timeofday"] != "daytime":
                continue

            self.img_fnames.append(os.path.join(self.img_dir, image["name"]))
            self.canny_fnames.append(os.path.join(self.canny_dir, image["name"]))
            self.time_of_day.append(image["attributes"]["timeofday"])

        print(f"Loaded {self.__len__()} data pairs.")

    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self, idx):
        img_fname = self.img_fnames[idx]
        canny_fname = self.canny_fnames[idx]
        time_of_day = self.time_of_day[idx]

        img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
        canny = cv2.imread(canny_fname)
        img = cv2.resize(img, (512, 512))
        canny = cv2.resize(canny, (512, 512))

        # Normalize image between [-1, 1]
        img = (img.astype(np.float32) / 127.5) - 1.0

        prompt = f"street at {time_of_day}"

        return dict(jpg=img, txt=prompt, hint=canny)


if __name__ == "__main__":
    dataset = BDD100K()
    item = dataset[1]
    jpg = item["jpg"]
    txt = item["txt"]
    hint = item["hint"]
    print(txt)
    print(jpg.shape)
    print(hint.shape)
