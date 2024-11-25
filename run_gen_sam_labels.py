import os
from PIL import Image
import torch
import torch.nn.functional as F
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


def semantic_sam_seg(image, model, remove_others=True):
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

    # stroke_inimg = None
    # stroke_refimg = None

    data = {"image": images, "height": height, "width": width}
    model.model.metadata = metadata
    batch_inputs = [data]
    results = model.model.evaluate(batch_inputs)
    pano_seg = results[-1]['panoptic_seg'][0]
    pano_seg_info = results[-1]['panoptic_seg'][1]
    new_pano_seg_info = []
    
    certain_pixels = torch.zeros(1).cuda()
    semantic_map = -1 * torch.ones_like(pano_seg)
    person_map = torch.zeros_like(pano_seg) # save the person instance maps
    idx = 1
    # 0: background, 1: person, -1: uncertain
    for i in range(len(pano_seg_info)):
        certain_pixels += (pano_seg == pano_seg_info[i]['id']).sum()
        if pano_seg_info[i]['category_id'] == 0:
            new_pano_seg_info.append(pano_seg_info[i])
            # person
            semantic_map[pano_seg == pano_seg_info[i]['id']] = 1
            person_map[pano_seg == pano_seg_info[i]['id']] = idx
            idx += 1
        else:
            # background: non person
            semantic_map[pano_seg == pano_seg_info[i]['id']] = 0
    certain_pixels = torch.clamp(certain_pixels, 1, w * h)


    demo = visual.draw_panoptic_seg(pano_seg.cpu(), new_pano_seg_info) # rgb Image
    res = demo.get_image()
    res = Image.fromarray(res)


    uncertain_ratio = 1 - (certain_pixels / (w * h))
    person_map[semantic_map == -1] = -1

    return len(new_pano_seg_info), res, uncertain_ratio, semantic_map, person_map


def count_with_sam_recursive(img, model, size=512, min_size=128, thr=0.2, remove_others=False):
    # first crop img to patches
    # then use sam to count
    # return a list of counts
    W, H = img.size
    if size == min_size or min(H, W) < size:
        check = False
    else:
        check = True
    counts = []
    
    W, H = img.size
    pred = Image.new('RGB', (W, H))
    semantic_map = torch.zeros((H, W))
    person_map = torch.zeros((H, W))
    for i in range(0, W, size):
        for j in range(0, H, size):
            # crop patch PIL image
            
            patch = img.crop((i, j, i+size, j+size))
            
            cnt, res, u_ratio, sem_map, p_map = semantic_sam_seg(patch, model, remove_others=remove_others)
            if check and u_ratio > thr:
                new_cnt, res, sem_map, p_map = count_with_sam_recursive(patch, model, size=size//2, thr=thr, remove_others=remove_others)
                cnt = max(cnt, new_cnt)
            res = res.resize((size, size))
            sem_map = F.interpolate(sem_map.float().unsqueeze(0).unsqueeze(0), size=(size, size), mode='nearest').squeeze(0).squeeze(0)
            p_map = F.interpolate(p_map.float().unsqueeze(0).unsqueeze(0), size=(size, size), mode='nearest').squeeze(0).squeeze(0)
            w, h = res.size
            if i + size > W and j + size > H:
                res = res.crop((0, 0, W-i, H-j))
                sem_map = sem_map[:H-j, :W-i]
                p_map = p_map[:H-j, :W-i]
            elif i + size > W:
                res = res.crop((0, 0, W-i, h))
                sem_map = sem_map[:, :W-i]
                p_map = p_map[:, :W-i]
            elif j + size > H:
                res = res.crop((0, 0, w, H-j))
                sem_map = sem_map[:H-j, :]
                p_map = p_map[:H-j, :]

            pred.paste(res, (i, j))

            w, h = res.size

            semantic_map[j:j+h, i:i+w] = sem_map
            p_map[p_map > 0] = p_map[p_map > 0] + person_map.max()
            person_map[j:j+h, i:i+w] = p_map
            counts.append(cnt)
    return np.array(counts).sum(), pred, semantic_map, person_map

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
opt['MODEL']['DECODER']['TEST']['OBJECT_MASK_THRESHOLD'] = 0.3

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

img_list = sorted(glob.glob('../../data/NWPU/train/*.jpg'))[::-1]
des_dir = f'../../data/NWPU_SAM/train_stage1_0/'


error = []

if not os.path.exists(des_dir):
    os.makedirs(des_dir)

N = len(img_list)
for i in range(N):
    img_path = img_list[i]
    des_path = os.path.join(des_dir, img_path.split('/')[-1].replace('.jpg', '.npy'))
    if os.path.exists(des_path):
        continue
    img = Image.open(img_path).convert('RGB')
    cnt, pred, sem_map, person_map = count_with_sam_recursive(img, model, size=512, thr=0.3, min_size=32)

    print(f'{i}/{N}', cnt, img.size)
    # save person map
    # downsample to 1/4 size
    person_map = F.interpolate(person_map.float().unsqueeze(0).unsqueeze(0), size=(img.size[1]//2, img.size[0]//2), mode='nearest').squeeze(0).squeeze(0)
    person_map = person_map.cpu().numpy().astype(np.int16)
    # save to disk
    np.save(des_path, person_map)
    
    