import cv2
from torchvision import transforms


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

class Transform_RGB():
    def __init__(self):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), 
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),  
                #transforms.Normalize(mean=[0.387, 0.482, 0.160], std=[0.076, 0.079, 0.071])]), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), 
            'valid': transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.387, 0.482, 0.160], std=[0.076, 0.079, 0.071])]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), 
            'test': transforms.Compose([
                transforms.ToTensor(),
               # transforms.Normalize(mean=[0.387, 0.482, 0.160], std=[0.076, 0.079, 0.071])])
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)



class Transform_LGB():
    def __init__(self):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), 
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),  
                transforms.Normalize(mean=[0.478, 0.482, 0.160], std=[0.078, 0.079, 0.071])]), 
            'valid': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.478, 0.482, 0.160], std=[0.078, 0.079, 0.071])]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.478, 0.482, 0.160], std=[0.078, 0.079, 0.071])])
        }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)
   

class Transform_LGB_tume():
    def __init__(self):
        self.data_transform = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), 
                # transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),  
                transforms.Normalize(mean=[0.478, 0.482, 0.160], std=[0.078, 0.079, 0.071])]), 
            'valid': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.478, 0.482, 0.160], std=[0.078, 0.079, 0.071])]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.478, 0.482, 0.160], std=[0.078, 0.079, 0.071])])
        }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


    
class Transform_RGL():
    def __init__(self):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), 
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),  
                transforms.Normalize(mean=[0.387, 0.482, 0.478], std=[0.076, 0.079, 0.078])]), 
            'valid': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.387, 0.482, 0.478], std=[0.076, 0.079, 0.078])]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.387, 0.482, 0.478], std=[0.076, 0.079, 0.078])])
        }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


class Transform_RLB():
    def __init__(self):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), 
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),  
                transforms.Normalize(mean=[0.387, 0.478, 0.160], std=[0.076, 0.078, 0.071])]), 
            'valid': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.387, 0.478, 0.160], std=[0.076, 0.078, 0.071])]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.387, 0.478, 0.160], std=[0.076, 0.078, 0.071])])
        }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)



class Transform_VGB():
    def __init__(self):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), 
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),  
                transforms.Normalize(mean=[0.482, 0.482, 0.160], std=[0.079, 0.079, 0.071])]), 
            'valid': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.482, 0.482, 0.160], std=[0.079, 0.079, 0.071])]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.482, 0.482, 0.160], std=[0.079, 0.079, 0.071])])
        }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


class Transform_RVB():
    def __init__(self):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), 
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),  
                transforms.Normalize(mean=[0.387, 0.482, 0.160], std=[0.076, 0.079, 0.071])]), 
            'valid': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.387, 0.482, 0.160], std=[0.076, 0.079, 0.071])]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.387, 0.482, 0.160], std=[0.076, 0.079, 0.071])])
        }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


class Transform_RGV():
    def __init__(self):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), 
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),  
                transforms.Normalize(mean=[0.387, 0.482, 0.482], std=[0.076, 0.079, 0.079])]), 
            'valid': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.387, 0.482, 0.482], std=[0.076, 0.079, 0.079])]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.387, 0.482, 0.482], std=[0.076, 0.079, 0.079])])
        }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)



def BGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def BGR2LAB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


def image_converter(img, type="LGB"):
    # tmp = img.copy()
    # cropped = tmp[int(H/2-112):int(H/2+112),int(W/2-112):int(W/2+112)]
    
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    RGB = cv2.split(RGB)
    LAB = cv2.split(LAB)
    HSV = cv2.split(HSV)

    if type == "RGB":
        return cv2.merge((RGB[0],RGB[1],RGB[2]))
    elif type == "LGB":
        return cv2.merge((LAB[0],RGB[1],RGB[2]))
    elif type == "RGL":
        return cv2.merge((RGB[0],RGB[1],RGB[2]))
    elif type == "RLB":
        return cv2.merge((RGB[0],LAB[0],RGB[2]))
    elif type == "VGB":
        return cv2.merge((HSV[0],RGB[1],RGB[2]))
    elif type == "RVB":
        return cv2.merge((RGB[0],HSV[2],RGB[2]))
    elif type == "RGV":
        return cv2.merge((RGB[0],RGB[1],HSV[2]))
    

def image_converter2(img,H,W,type="LGB"):
    # tmp = img.copy()
    # cropped = tmp[int(H/2-112):int(H/2+112),int(W/2-112):int(W/2+112)]
    cropped = img
    RGB = BGR2RGB(cropped.copy())
    RGB = cv2.split(RGB)
    LAB = BGR2LAB(cropped.copy())
    LAB = cv2.split(LAB)

    HSV = cv2.cvtColor(cropped.copy(), cv2.COLOR_BGR2HSV)
    HSV = cv2.split(HSV)

    if type == "RGB":
        return cv2.merge((RGB[0],RGB[1],RGB[2]))
    elif type == "LGB":
        return cv2.merge((LAB[0],RGB[1],RGB[2]))
    elif type == "RGL":
        return cv2.merge((RGB[0],RGB[1],RGB[2]))
    elif type == "RLB":
        return cv2.merge((RGB[0],LAB[0],RGB[2]))
    elif type == "VGB":
        return cv2.merge((HSV[0],RGB[1],RGB[2]))
    elif type == "RVB":
        return cv2.merge((RGB[0],HSV[2],RGB[2]))
    elif type == "RGV":
        return cv2.merge((RGB[0],RGB[1],HSV[2]))



image_transform = {
    "RGB":Transform_RGB,
    "LGB":Transform_LGB,
    "RGL":Transform_RGL,
    "RLB":Transform_RLB,
    "VGB":Transform_VGB,
    "RVB":Transform_RVB,
    "RGV":Transform_RGV,
    "LGB_tume":Transform_LGB_tume,
}
