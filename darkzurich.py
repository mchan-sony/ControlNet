import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from glob import glob
from annotator.util import resize_image, HWC3


model_canny = None

# def process(img, res=256, l=100, h=200): cv2
    # pass

def canny(img, res, l, h):
    img = resize_image(HWC3(img), res)
    global model_canny
    if model_canny is None:
        from annotator.canny import CannyDetector
        model_canny = CannyDetector()
    result = model_canny(img, l, h)
    return [result]


class DarkZurich(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

if __name__ == "__main__":
    fnames = glob('/data/matthew/dark_zurich/train/rgb_anon/train/night/**/*.png')
    img = cv2.imread(fnames[600])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256,256))
    # out = canny(img, 256, 100, 200)
    # out = cv2.Canny(img, 100, 200, 7, L2gradient=True)
    out = cv2.Canny(img, 50, 100, 7, L2gradient=True)
    print(out.shape)
    cv2.imwrite("canny.png", out)