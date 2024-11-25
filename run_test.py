import os
from PIL import Image
import sys
import torch
import argparse
import whisper
import numpy as np
import glob

from torchvision import transforms
from detectron2.data import MetadataCatalog
from utils.visualizer import Visualizer

from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES


def semantic_sam_seg(image, remove_others=True):
    h, w = image.size
    metadata = MetadataCatalog.get('coco_2017_train_panoptic')
    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)
    image_ori = transform(image)
    width = image_ori.size[0]
    height = image_ori.size[1]
    image_ori = np.asarray(image_ori)
    visual = Visualizer(image_ori, metadata=metadata)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()


    data = {"image": images, "height": height, "width": width}
    model.model.metadata = metadata
    batch_inputs = [data]
    results = model.model.evaluate(batch_inputs)
    pano_seg = results[-1]['panoptic_seg'][0]
    pano_seg_info = results[-1]['panoptic_seg'][1]
    new_pano_seg_info = []
    
    certain_pixels = torch.zeros(1).cuda()
    for i in range(len(pano_seg_info)):
        certain_pixels += (pano_seg == pano_seg_info[i]['id']).sum()
        if pano_seg_info[i]['category_id'] == 0:
            # print(pano_seg_info[i]['isthing'])
            new_pano_seg_info.append(pano_seg_info[i])
        elif not remove_others:
            new_pano_seg_info.append(pano_seg_info[i])
    certain_pixels = torch.clamp(certain_pixels, 1, w * h)


    demo = visual.draw_panoptic_seg(pano_seg.cpu(), new_pano_seg_info) # rgb Image
    res = demo.get_image()
    res = Image.fromarray(res)


    uncertain_ratio = 1 - (certain_pixels / (w * h))

    return len(new_pano_seg_info), res, uncertain_ratio

def count_with_sam(img, size=256, remove_others=True):
    # first crop img to patches
    # then use sam to count
    # return a list of counts
    size = size
    stride = size
    counts = []
    # resize img to the 256xN
    H, W = img.size
    img = img.resize((H // size * size, W // size * size), resample=Image.BICUBIC)
    H, W = img.size
    pred = Image.new('RGB', (H, W))
    for i in range(0, H, stride):
        for j in range(0, W, stride):
            # crop patch PIL image
            patch = img.crop((i, j, i+size, j+size))
            # patch = img[i:i+size, j:j+size, :]
            cnt, res, u_ratio = semantic_sam_seg(patch, remove_others=remove_others)
            res = res.resize((size, size))
            pred.paste(res, (i, j))
            counts.append(cnt)
            # print(i, j, cnt, u_ratio)
    return np.array(counts).sum(), pred

def count_with_sam_recursive(img, size=512, min_size=128, thr=0.2, remove_others=True):
    # first crop img to patches
    # then use sam to count
    # return a list of counts
    H, W = img.size
    if size == min_size or min(H, W) < size:
        return count_with_sam(img, size=min_size, remove_others=remove_others)
    else:
        counts = []
        
        H, W = img.size
        pred = Image.new('RGB', (H, W))
        for i in range(0, H, size):
            for j in range(0, W, size):
                # crop patch PIL image
                patch = img.crop((i, j, i+size, j+size))
                cnt, res, u_ratio = semantic_sam_seg(patch, remove_others=remove_others)
                if u_ratio > thr:
                    new_cnt, res = count_with_sam_recursive(patch, size=size//2, thr=thr, remove_others=remove_others)
                    cnt = max(cnt, new_cnt)
                res = res.resize((size, size))
                pred.paste(res, (i, j))
                counts.append(cnt)
                # print(i, j, cnt, u_ratio)
    return np.array(counts).sum(), pred

def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/seem/focall_unicl_lang_demo.yaml", metavar="FILE", help='path to config file', )
    cfg = parser.parse_args('')
    return cfg

'''
build args
'''
cfg = parse_option()
opt = load_opt_from_config_files([cfg.conf_files])
opt = init_distributed(opt)


# META DATA
cur_model = 'None'
if 'focalt' in cfg.conf_files:
    pretrained_pth = os.path.join("seem_focalt_v0.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v0.pt"))
    cur_model = 'Focal-T'
elif 'focal' in cfg.conf_files:
    pretrained_pth = os.path.join("seem_focall_v0.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt"))
    cur_model = 'Focal-L'

'''
build model
'''
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

audio = whisper.load_model("base")

size = int(sys.argv[1])
min_size = int(sys.argv[2])
thr = float(sys.argv[3])


# 283, 488 
img_list = sorted(glob.glob('../../data/UCF/test/*.jpg'))
N = len(img_list)

# for each image, read and segment
error = []
# init a file to record error
log_file = f'logs/log_{size}_{min_size}_{thr}.txt'
with open(log_file, 'w') as f:
    f.write('')
for i in range(N):
    img_path = img_list[i]
    lab_path = img_path.replace('.jpg', '.npy')
    label = np.load(lab_path)
    img = Image.open(img_path).convert('RGB')
    cnt, _ = count_with_sam_recursive(img, size=size, thr=thr, min_size=min_size)
    error.append(cnt - len(label))
    mae = np.mean(np.abs(error))
    mse = np.sqrt(np.mean(np.square(error)))

    print(f'{i}/{N}', cnt - len(label), cnt, len(label), mae, mse)
    with open(log_file, 'a') as f:
        f.write(f'{i}/{N} {cnt - len(label)} {cnt} {len(label)} {mae} {mse}\n')