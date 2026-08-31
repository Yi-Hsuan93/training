import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
import os
import pandas as pd

class OurDataset(torch.utils.data.Dataset):

  def __init__(self, root_path: str, df: pd.DataFrame, is_train: bool = True):
    self.data_infos = []
        
    # 逐行把表格裡面的資料拿出來
    for index, row in df.iterrows():
      img_name = row['ID']    # 直接用欄位名稱拿資料，超直覺！

    # 【防呆機制】：如果 Label 是空的 (NaN)，就先暫時給它 0，避免 int() 報錯
      if pd.isna(row['Label']):
          label = 0
      else:
        label = int(row['Label'])
            
      img_path = os.path.join(root_path, img_name)
      self.data_infos.append({"path": img_path, "label": label})

    # print(self.data_infos[3])
    # 【切換模式】：依據是不是訓練階段，給予不同的前處理
      if is_train:
        self.trans = transforms.Compose([   # transforms 是一個「模組 (Module)」、Compose 是放在transforms的「類別 (Class)」
          transforms.ToPILImage(),             # 1. 將 cv2 的 numpy array 轉為 PIL Image
          transforms.Resize((224, 224)),       # 2. 統一縮放到固定大小
                
          # --- Data Augmentation 區塊開始 ---
          transforms.RandomHorizontalFlip(p=0.5), # 以 50% 的機率水平翻轉圖片
          transforms.Pad(padding=40, padding_mode='reflect'), # 1. 先向外鏡像延伸 40 像素
          transforms.RandomRotation(degrees=15),              # 2. 旋轉圖片
          transforms.CenterCrop(224),                         # 3. 把圖片切回 224x224
          transforms.ColorJitter(brightness=0.2, contrast=0.2), # 隨機微調亮度與對比
          # --- Data Augmentation 區塊結束 ---
                
          transforms.ToTensor(),               # 3. 轉成 Tensor 並將數值縮放到 0~1
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 4. 標準化到 -1~1
        ])
      else:
        self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                # --- 測試時只需純粹的轉換與正規化 ---
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
  def __len__(self):
    return len(self.data_infos)

  def __getitem__(self, index: int):
    img_path = self.data_infos[index]["path"]
    data = cv2.imread(img_path)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB) # BGR -> RGB

    # print(f"before transform shape: {data.shape}")
    data = self.trans(data)
    # print(f"after transform shape: {data.shape}")

    label = self.data_infos[index]["label"]

    return data, label


if __name__ == "__main__":
  test_df = pd.read_csv('./data/train.csv')
  dataset = OurDataset(root_path = './data/train_images', df = test_df)
  train_loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=4, shuffle=False)

  for images, labels in train_loader:
    print(f"images : ", {images})
    print(f"labels : ", {labels})
    print(f"images shape: ", {images.shape})
    print(f"labels shape: ", {labels.shape})
    break