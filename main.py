import pandas as pd
import time
import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model import build_model
from dataset import OurDataset

MAX_EPOCH = 40
BATCH_SIZE = 16
LR = 1e-4           
MODEL_NAME = 'resnet50.tv_in1k'
NUMBER_CLASSES = 6
ROOT_CSV = './data/train.csv'
NUM_WORKERS = 8 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 修改 1：將 scaler 作為參數傳入 train 函式 =================
def train(loader, optimizer, model, criterion, scaler):
    model.train()
    
    train_loss = 0.0
    correct = 0
    total = 0

    for index, (image, label) in enumerate(tqdm(loader, desc="Training")):
        image, label = image.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        
        # ================= 修改 2：開啟 autocast (自動混合精度) =================
        # 告訴 PyTorch 這個區塊內的正向傳播 (Forward) 可以用 float16 加速
        with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda')):
            output = model(image)
            loss = criterion(output, label)

        # ================= 修改 3：透過 scaler 處理反向傳播 =================
        # 因為 float16 的數值範圍很小，scaler 會自動放大 Loss，避免梯度下溢 (Underflow) 變成 0
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        
        predictions = torch.argmax(output, dim=1)
        correct += (predictions == label).sum().item()
        total += label.size(0)

    avg_loss = train_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def val(loader, model, criterion):
    model.eval()
    
    val_loss = 0.0
    correct = 0
    total = 0
    
    for index, (image, label) in enumerate(tqdm(loader, desc="Validation")):
        image, label = image.to(DEVICE), label.to(DEVICE)

        with torch.no_grad():
            # 驗證集同樣開啟 autocast，這能大幅加快推論速度並節省記憶體
            with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda')):
                output = model(image)
                loss = criterion(output, label)
            
            val_loss += loss.item()
            
            predictions = torch.argmax(output, dim=1)
            correct += (predictions == label).sum().item()
            total += label.size(0)

    avg_loss = val_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    df = pd.read_csv(ROOT_CSV)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    train_set = OurDataset(root_path='./data/train_images', df=train_df, is_train=True)
    val_set = OurDataset(root_path='./data/train_images', df=val_df, is_train=False)

    print(f"訓練集資料筆數: {len(train_set)}")
    print(f"驗證集資料筆數: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model(model_name=MODEL_NAME, num_classes=NUMBER_CLASSES)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    
    # ================= 修改 4：在主要迴圈前宣告 GradScaler =================
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))

    best_val_loss = float('inf') 
    val_interval = 5  

    # 在準備進入訓練迴圈前，按下碼錶，記錄開始時間
    print("開始訓練...")
    start_time = time.time()
    
    for epoch in range(MAX_EPOCH):
        print(f"\n========== Epoch {epoch+1}/{MAX_EPOCH} ==========")
        
        # ================= 修改 5：記得把 scaler 傳進去 =================
        train_loss, train_acc = train(train_loader, optimizer, model, criterion, scaler)
        print(f"[Train] Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")

        if (epoch + 1) % val_interval == 0 or (epoch + 1) == MAX_EPOCH:
            val_loss, val_acc = val(val_loader, model, criterion)
            print(f"[Valid] Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"🌟 發現更好的模型 (Val Loss 降至 {best_val_loss:.4f})，已儲存 best_model.pth！")

    # 計算總訓練時間
    end_time = time.time()
    print(f"訓練完成！總耗時: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()