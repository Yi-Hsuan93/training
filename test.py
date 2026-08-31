import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import build_model
from dataset import OurDataset

# --- 參數設定 ---
MODEL_NAME = 'resnet50.tv_in1k'
NUMBER_CLASSES = 6
BATCH_SIZE = 16
NUM_WORKERS = 8 # WSL 環境若報錯可改為 0 或 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_CSV_PATH = './data/test.csv'         # 測試集 CSV 路徑
TEST_IMG_DIR = './data/test_images'       # 測試集圖片資料夾路徑

def main():
    print(f"1. 讀取 {TEST_CSV_PATH}...")
    test_df = pd.read_csv(TEST_CSV_PATH)

    print("2. 準備測試集 DataLoader...")
    # 【關鍵】：這裡傳入 is_train=False，確保不會做隨機變形，並且 shuffle=False 確保順序與 CSV 一致
    test_set = OurDataset(root_path=TEST_IMG_DIR, df=test_df, is_train=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("3. 建立模型並載入最強權重 best_model.pth...")
    model = build_model(model_name=MODEL_NAME, num_classes=NUMBER_CLASSES)
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(DEVICE)
    model.eval() # 進入評估模式

    print("4. 開始進行測試集推論...")
    all_predictions = []

    with torch.no_grad():
        # 用底線 _ 忽略 dataset 回傳的假標籤
        for image, _ in tqdm(test_loader, desc="Testing"):
            image = image.to(DEVICE)
            
            output = model(image)
            predictions = torch.argmax(output, dim=1)
            
            # 將結果轉回 CPU 並收集起來
            all_predictions.extend(predictions.cpu().tolist())

    print("5. 將預測結果寫回 CSV 檔...")
    # 把我們預測出來的陣列，直接覆蓋掉 DataFrame 裡面的 Label 欄位
    test_df['Label'] = all_predictions

    # 存檔成 submission.csv 準備上傳或提交
    output_filename = 'submission.csv'
    test_df.to_csv(output_filename, index=False)
    
    print(f"🎉 測試集預測完成！所有結果已成功儲存至 {output_filename}")

if __name__ == "__main__":
    main()