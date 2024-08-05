import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
import tifffile
import re
import cv2 as cv

def load_image(f_img):
    img_dict = dict()
    with tifffile.TiffFile(f_img) as tif:
        for page in tif.pages:
            page_desc = page.description
            if page_desc.find("FullResolution") == -1:
                continue
            name_li = list(set(re.findall("<Biomarker>([ \S]+)</Biomarker>", page.description)))
            if len(name_li)==0:
                continue
            name = name_li[0]
            img_dict[name] = (page.asarray(), page.description)
    return img_dict


def denoise_by_Noise2Fast(img, device, learning_rate = 0.001, tsince = 100):
    """https://github.com/jason-lequyer/Noise2Fast/blob/main/N2F.py"""
    class TwoCon(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            return x

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = TwoCon(1, 64)
            self.conv2 = TwoCon(64, 64)
            self.conv3 = TwoCon(64, 64)
            self.conv4 = TwoCon(64, 64)  
            self.conv6 = nn.Conv2d(64,1,1)
            
        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            x = self.conv4(x3)
            x = torch.sigmoid(self.conv6(x))
            return x
    
    notdone = True
    while notdone:
        img = img.astype(np.float32)   
        minner = np.amin(img)
        img = img - minner
        maxer = np.amax(img)
        img = img/maxer
        img = img.astype(np.float32)
        shape = img.shape
        
        listimgH = []
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
            Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1]]
        
        imgin = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        imgin2 = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        for i in range(imgin.shape[0]):
            for j in range(imgin.shape[1]):
                if j % 2 == 0:
                    imgin[i,j] = imgZ[2*i+1,j]
                    imgin2[i,j] = imgZ[2*i,j]
                if j % 2 == 1:
                    imgin[i,j] = imgZ[2*i,j]
                    imgin2[i,j] = imgZ[2*i+1,j]
        imgin = torch.from_numpy(imgin)
        imgin = torch.unsqueeze(imgin,0)
        imgin = torch.unsqueeze(imgin,0)
        imgin = imgin.to(device)
        imgin2 = torch.from_numpy(imgin2)
        imgin2 = torch.unsqueeze(imgin2,0)
        imgin2 = torch.unsqueeze(imgin2,0)
        imgin2 = imgin2.to(device)
        listimgH.append(imgin)
        listimgH.append(imgin2)
        
        listimgV = []
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
            Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1]]
        
        imgin3 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        imgin4 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        for i in range(imgin3.shape[0]):
            for j in range(imgin3.shape[1]):
                if i % 2 == 0:
                    imgin3[i,j] = imgZ[i,2*j+1]
                    imgin4[i,j] = imgZ[i, 2*j]
                if i % 2 == 1:
                    imgin3[i,j] = imgZ[i,2*j]
                    imgin4[i,j] = imgZ[i,2*j+1]
        imgin3 = torch.from_numpy(imgin3)
        imgin3 = torch.unsqueeze(imgin3,0)
        imgin3 = torch.unsqueeze(imgin3,0)
        imgin3 = imgin3.to(device)
        imgin4 = torch.from_numpy(imgin4)
        imgin4 = torch.unsqueeze(imgin4,0)
        imgin4 = torch.unsqueeze(imgin4,0)
        imgin4 = imgin4.to(device)
        listimgV.append(imgin3)
        listimgV.append(imgin4)
        
    
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img,0)
        img = torch.unsqueeze(img,0)
        img = img.to(device)
        
        listimgV1 = [[listimgV[0],listimgV[1]]]
        listimgV2 = [[listimgV[1],listimgV[0]]]
        listimgH1 = [[listimgH[1],listimgH[0]]]
        listimgH2 = [[listimgH[0],listimgH[1]]]
        listimg = listimgH1+listimgH2+listimgV1+listimgV2
        
        net = Net()
        net.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        
        running_loss1=0.0
        running_loss2=0.0
        maxpsnr = -np.inf
        timesince = 0
        last10 = [0]*105
        last10psnr = [0]*105
        cleaned = 0
        while timesince <= tsince:
            indx = np.random.randint(0,len(listimg))
            data = listimg[indx]
            inputs = data[0]
            labello = data[1]
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss1 = criterion(outputs, labello)
            loss = loss1
            running_loss1+=loss1.item()
            loss.backward()
            optimizer.step()
            
            running_loss1=0.0
            with torch.no_grad():
                last10.pop(0)
                last10.append(cleaned*maxer+minner)
                outputstest = net(img)
                cleaned = outputstest[0,0,:,:].cpu().detach().numpy()
                
                noisy = img.cpu().detach().numpy()
                ps = -np.mean((noisy-cleaned)**2)
                last10psnr.pop(0)
                last10psnr.append(ps)
                if ps > maxpsnr:
                    maxpsnr = ps
                    timesince = 0
                else:
                    timesince+=1.0
        H = np.mean(last10, axis=0)
        if np.sum(np.round(H[1:-1,1:-1]-np.mean(H[1:-1,1:-1]))>0) <= 25 and learning_rate != 0.000005:
            learning_rate = 0.000005
            print("Reducing learning rate")
        else:
            notdone = False
    torch.cuda.empty_cache()
    return H


def denoise_by_rescale(img_arr, x_range, y_range):
    outer_region = img_arr.copy()
    outer_region[x_range[0]:x_range[1], y_range[0]:y_range[1]] = 0
    nzo_outer_region = outer_region[outer_region>0]
    cutoff = np.median(nzo_outer_region) * 2
    center_region = img_arr[x_range[0]:x_range[1], y_range[0]:y_range[1]]
    center_region[center_region<cutoff] = 0
    center_region = center_region.astype(np.float32)
    high_value_cutoff = np.percentile(center_region, 99.95)
    center_region[center_region > high_value_cutoff] = high_value_cutoff
    center_region = center_region - center_region.min()
    center_region = center_region / center_region.max()
    return center_region


def IF_denoise_main(args):
    img_dict = load_image(args.image)
    DAPI_img_arr = img_dict["DAPI"][0]
    nzo_val = DAPI_img_arr[DAPI_img_arr>0].reshape([-1,1])
    gmm = GaussianMixture(n_components=2).fit(nzo_val)
    cutoff = np.mean(gmm.means_)
    mask_pos_arr = DAPI_img_arr>cutoff
    signal_x_pos, signal_y_pos = np.where(mask_pos_arr)
    x_range = np.percentile(signal_x_pos, np.array([0.5, 99.5])).astype(int)
    y_range = np.percentile(signal_y_pos, np.array([0.5, 99.5])).astype(int)
    

    with tifffile.TiffWriter(args.output, bigtiff=True) as tif:
        for gid, (img_arr, description) in img_dict.items():
            if args.method == "Noise2Fast":
                crop_img_arr = img_arr[x_range[0]:x_range[1], y_range[0]:y_range[1]]
                denoise_crop_img_arr = denoise_by_Noise2Fast(crop_img_arr, "cpu")
            elif args.method == "rescale":
                denoise_crop_img_arr = denoise_by_rescale(img_arr, mask_pos_arr)
            else:
                raise ValueError()
            denoise_img_arr = np.zeros_like(img_arr, dtype=np.float32)
            denoise_img_arr[x_range[0]:x_range[1], y_range[0]:y_range[1]] = denoise_crop_img_arr
            denoise_img_arr = (255 * denoise_img_arr / denoise_img_arr.max()).astype(np.uint8)
            tif.write(denoise_img_arr, description=description)
