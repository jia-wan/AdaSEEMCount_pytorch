from PIL import Image

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import glob
import time
from scipy.spatial import distance

from torchvision import transforms



import sys
sys.path.append('../novel_loss/')
from models.vgg import vgg19


# define a new collate function
def cc_collate_fn(batch):
    # image and lab can use default, points need to be a list
    images, labs, points = [], [], []
    for sample in batch:
        images.append(sample[0])
        labs.append(sample[1])
        points.append(sample[2])
    images = torch.stack(images, dim=0)
    labs = torch.stack(labs, dim=0)
    return images, labs, points

# define a dataloader load image and label
class SAMCrowdDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, stage='train', dataset='ucf'):
        
        self.stage = stage
        if 'train' in stage:
            if 'sha' in dataset:
                lab_list = sorted(glob.glob('../../data/ShanghaiTech_SAM/part_A/train_stage4_1/*.npy'))
                img_list = [x.replace('npy', 'jpg').replace('ShanghaiTech_SAM/part_A/train_stage4_1', 'ShanghaiTech/part_A/train_data/images') for x in lab_list]
            elif 'shb' in dataset:
                lab_list = sorted(glob.glob('../../data/ShanghaiTech_SAM/part_B/train_stage3_1/*.npy'))
                img_list = [x.replace('npy', 'jpg').replace('ShanghaiTech_SAM/part_B/train_stage3_1', 'ShanghaiTech/part_B/train_data/images') for x in lab_list]
            elif 'jhu' in dataset:
                lab_list = sorted(glob.glob('../../data/jhu_SAM/train_stage3_1/*.npy'))
                img_list = [x.replace('npy', 'jpg').replace('jhu_SAM/train_stage3_1', 'jhu_crowd_v2.0/train/images') for x in lab_list]
            elif 'nwpu' in dataset:
                lab_list = sorted(glob.glob('../../data/NWPU_SAM/train_stage3_1/*.npy'))
                img_list = [x.replace('npy', 'jpg').replace('NWPU_SAM/train_stage3_1', 'NWPU/train') for x in lab_list]
            else:
                lab_list = sorted(glob.glob('../../data/UCF_SAM/train_stage3_1/*.npy'))
                img_list = [x.replace('npy', 'jpg').replace('UCF_SAM/train_stage3_1', 'UCF-QNRF_ECCV18/Train') for x in lab_list]
        elif 'val' in stage:
            if 'jhu' in dataset:
                lab_list = sorted(glob.glob('../../data/jhu_crowd_v2.0/val/images/*.npy'))
                img_list = [x.replace('npy', 'jpg') for x in lab_list]
            else:
                lab_list = sorted(glob.glob('../../data/UCF_Bayes/val/*.npy'))
                img_list = [x.replace('npy', 'jpg').replace('UCF_Bayes/val', 'UCF-QNRF_ECCV18/Train') for x in lab_list]
        else:
            if 'sha' in dataset:
                lab_list = sorted(glob.glob('../../data/ShanghaiTech/part_A/test_data/images/*.npy'))
                img_list = [x.replace('npy', 'jpg') for x in lab_list]
            elif 'shb' in dataset:
                lab_list = sorted(glob.glob('../../data/ShanghaiTech/part_B/test_data/images/*.npy'))
                img_list = [x.replace('npy', 'jpg') for x in lab_list]
            elif 'jhu' in dataset:
                lab_list = sorted(glob.glob('../../data/jhu_crowd_v2.0/test/images/*.npy'))
                img_list = [x.replace('npy', 'jpg') for x in lab_list]
            elif 'nwpu' in dataset:
                lab_list = sorted(glob.glob('../../data/NWPU/test/*.npy'))
                img_list = [x.replace('npy', 'jpg') for x in lab_list]
            else:
                lab_list = sorted(glob.glob('../../data/UCF_Bayes/test/*.npy'))
                img_list = [x.replace('npy', 'jpg').replace('UCF_Bayes/test', 'UCF-QNRF_ECCV18/Test') for x in lab_list]
        
        self.img_list = img_list
        self.lab_list = lab_list
        self.transform = transform

    def __len__(self):
        if 'train' in self.stage:
            return len(self.img_list) * 1
        return len(self.img_list)
    
    def __getitem__(self, idx):
        idx = idx % len(self.img_list)
        img = Image.open(self.img_list[idx]).convert('RGB')
        # lab = np.load(self.lab_list[idx])
        if False: #'train' in self.stage:
            data = np.load(self.lab_list[idx], allow_pickle=True).item()
            lab = data['person_map']
            points = data['all_points']
        else:
            lab = np.load(self.lab_list[idx])
        # upsample the lab to the same size as img
        if 'train' in self.stage:
            # resize image if min size < 512
            if min(img.size) < 512:
                # min size = 512 keep the aspect ratio
                w, h = img.size
                if w < h:
                    img = img.resize((512, int(512 * h / w)), resample=Image.BICUBIC)
                    lab = cv2.resize(lab, (img.size[0], img.size[1]), interpolation=cv2.INTER_NEAREST)
                else:
                    img = img.resize((int(512 * w / h), 512), resample=Image.BICUBIC)
                    lab = cv2.resize(lab, (img.size[0], img.size[1]), interpolation=cv2.INTER_NEAREST)
            lab = cv2.resize(lab, (img.size[0], img.size[1]), interpolation=cv2.INTER_NEAREST)
            # random crop 512 x 512 patch image and label
            w, h = img.size
            i = 0 if w - 512 == 0 else np.random.randint(0, w - 512)
            j = 0 if h - 512 == 0 else np.random.randint(0, h - 512)
            img = img.crop((i, j, i+512, j+512))
            lab = lab[j:j+512, i:i+512]
            lab = lab[np.newaxis, :, :]

        if self.transform:
            img = self.transform(img)
        lab = torch.from_numpy(lab)

        return img, lab

def localize_head(mask, ratio=0.9):
    """
    Given a binary mask of a person, localize the head center.
    We assume the head center is about 15% of the total hight.
    """
    # y is 15% of the height
    # bounding box of nonzero elements in the mask
    points = np.argwhere(mask > 0)
    x, y, w, h = points[:, 1].min(), points[:, 0].min(), points[:, 1].max() - points[:, 1].min(), points[:, 0].max() - points[:, 0].min()
    # Compute the height of the mask
    y = y + h * (1 - ratio)
    row_y = mask[int(y), :]
    # Find the middle nonzero element
    mid = np.argwhere(row_y > 0)
    if len(mid) == 0:
        return np.array([x + w // 2, y]).astype(np.int32)
    else:
        _min = mid.min()
        _max = mid.max()
        mid = (_min + _max) // 2
    return np.array([mid, y]).astype(np.int32)

def region_mae_loss(pred, label, points=None, weight=1, bkg_weight=1, ratio=0.9):
    """
    Calculate the error between the predicted region and the ground truth region for each person id in the label map.
    """
    B, C, H, W = pred.size()
    # resize label to the same size as pred
    label = F.interpolate(label, size=(pred.size(2), pred.size(3)), mode='nearest')
    loss = torch.mean(torch.abs(pred - pred))
    # backgrond should be zero
    if (label == 0).sum() > 0:
        loss += bkg_weight * (pred[label == 0]).sum()
    # if torch.isnan(loss):
    #     print('background isnan', loss, pred[label == 0])
    # set device as pred device
    device = pred.device
    for b in range(B):
        for person_id in torch.unique(label[b]):
            if person_id <= 0:
                continue
            mask = (label[b] == person_id).float()
            region_pred = pred[b] * mask
            # normalize region pred and focus on center
            # print(localize_head(mask[0].cpu().numpy()), points[b][int(person_id)])
            if points is None:
                center = localize_head(mask[0].cpu().numpy(), ratio=ratio)
            else:
                center = points[b][int(person_id)] / 8
                if min(center) < 0 or max(center) > 512:
                    center = localize_head(mask[0].cpu().numpy(), ratio=ratio)
            # the prediction far from the center should be smaller
            cost = distance.cdist(np.argwhere(mask[0].cpu().numpy() > 0)[:,::-1] / 512, center.reshape(1, 2) / 512, 'euclidean')
            cost = np.exp(cost / 0.6) - 1
            
            cost = torch.from_numpy(cost).float().to(device)
            transport = pred[b][0][mask[0].cpu().numpy() > 0]
            # normalize transport, sum to 1
            transport = transport / (transport.sum() + 1e-8)
            total_cost = (cost.reshape(-1) * transport.reshape(-1)).sum()
            mae = (torch.abs(region_pred.sum() - 1)).sum()

            loss += (mae + weight * total_cost)

    
    return loss / B

def evaluate(model, data_loader):
    model.eval()
    error = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):

            data, target = data.cuda(), target.cuda()
            # if data shape > 2048, split into 1024 patches
            if data.shape[2] > 2048 or data.shape[3] > 2048:
                pred_num = 0
                for i in range(0, data.shape[2] - 1024, 1024):
                    for j in range(0, data.shape[3] - 1024, 1024):
                        if i + 2048 > data.shape[2] and j + 2048 > data.shape[3]:
                            data_patch = data[:, :, i:, j:]
                        elif i + 2048 > data.shape[2]:
                            data_patch = data[:, :, i:, j:j+1024]
                        elif j + 2048 > data.shape[3]:
                            data_patch = data[:, :, i:i+1024, j:]
                        else:
                            # data_patch = data[:, :, i:, j:]
                            data_patch = data[:, :, i:i+1024, j:j+1024]
                        output, _ = model(data_patch)
                        pred_num += output.sum().item()
            else:
                output, _ = model(data)
                pred_num = output.sum().item()
            error.append(pred_num - len(target[0]))
    mae = np.mean(np.abs(error))
    mse = np.sqrt(np.mean(np.square(error)))
    return mae, mse

def validate(model, data_loader):
    model.eval()
    error = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # count the number of people without uncertain region
            data, target = data.cuda(), target.cuda()
            # print(batch_idx, data.shape)        
            output, _ = model(data)
            # reszie target 
            target = F.interpolate(target.float(), size=(output.size(2), output.size(3)), mode='nearest')
            mask = target >= 0
            output = output * mask
            error.append( output.sum().item() - (len(torch.unique(target[0])) -2) )
    mae = np.mean(np.abs(error))
    mse = np.sqrt(np.mean(np.square(error)))
    return mae, mse

def train_epoch(model, optimizer, criterion, data_loader, weight=0.1, bkg_weight=1, ratio=0.9):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.float().cuda()
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, target, weight=weight, bkg_weight=bkg_weight, ratio=ratio)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(data_loader)


# define tranform for ImageNet pretrained model
# normalize tranform ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225], inplace=True)
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

dataset_name = sys.argv[1] #'jhu
weight = float(sys.argv[2]) # 100
ratio = float(sys.argv[3]) # 0.9
bkg_weight = float(sys.argv[4]) # 1
momentum = float(sys.argv[5]) # 0.7

train_dataset = SAMCrowdDataset(transform=transform, stage='train', dataset=dataset_name)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)

val_dataset = SAMCrowdDataset(transform=transform, stage='val', dataset=dataset_name)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)


model = vgg19()
model = model.cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# define a log file to save the training log
log_path = f'logs/1219_stable_{dataset_name}_stage3_{weight}_bkg{bkg_weight}_{ratio}_b1_m{momentum}.txt'
val_path = log_path.replace('.txt', '_val_best.pth')
with open(log_path, 'w') as f:
    f.write('logs\n')

# write params and time
with open(log_path, 'a') as f:
    f.write(f'weight: {weight}, bkg_weight: {bkg_weight}, ratio: {ratio}\n')
    f.write(f'params: {sum(p.numel() for p in model.parameters())}\n')
    f.write(f'time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')

best_mae = 1000
best_mse = 1000
best_epoch = 0

# train the model
num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train_epoch(model, optimizer, region_mae_loss, train_loader, weight=weight, bkg_weight=bkg_weight, ratio=ratio)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
    with open(log_path, 'a') as f:
        f.write(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}\n")
    if (epoch + 1) % 1 == 0:

        mae, mse = evaluate(model, val_loader)
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_epoch = epoch + 1
            torch.save(model.state_dict(), val_path)
       
        print(f"Epoch {epoch+1}, Val MAE: {mae:.4f}, Val MSE: {mse:.4f}")
        print(f"Best Val MAE: {best_mae:.4f}, Best Val MSE: {best_mse:.4f}, Best Val Epoch: {best_epoch}")
        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch+1}, Val MAE: {mae:.4f}, Val MSE: {mse:.4f}\n")
            f.write(f"Best Val MAE: {best_mae:.4f}, Best Val MSE: {best_mse:.4f}, Best Val Epoch: {best_epoch}\n")
            # write time
            f.write(f'time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')