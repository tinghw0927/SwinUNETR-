# SwinUNETR-
AI CUP 2025 SwinUNETR
# AI Cup 2025 - 心臟CT影像之心臟結構分割競賽

本專案使用 SwinUNETR 模型進行3D心臟CT影像的多類別語義分割。

## 📋 競賽資訊
- 競賽名稱：AI Cup 2025 - 心臟CT影像之心臟結構分割
- 使用模型：SwinUNETR
- 最終成績：Private Leaderboard Dice Score = [你的分數]

## 🔧 環境需求

### 硬體
- GPU: NVIDIA Tesla T4 (16GB VRAM) 或以上
- RAM: 12GB 以上

### 軟體
- Python 3.10
- Google Colab Pro (推薦)

### 套件安裝
```bash
pip install monai[all]
pip install torch torchvision
pip install nibabel
pip install scipy
pip install matplotlib
```

## 📁 資料準備

### 資料集結構
請將資料集放置於以下結構：
```
/content/drive/MyDrive/aicup_data/
├── train/
│   ├── imagesTr/
│   └── labelsTr/
└── test/
    └── imagesTs/
```

## 🚀 使用方法

### 訓練模型
1. 開啟 Colab notebook
2. 連接 Google Drive
3. 執行所有程式碼區塊
4. 模型會自動訓練並儲存權重

### 預測
訓練完成後，執行推論部分的程式碼即可生成預測結果。

## 🎯 模型架構
- **模型**: SwinUNETR
- **輸入大小**: 128*128*128
- **類別數**: 7類（LV, RV, LA, RA, LAD, LCX, RCA）
- **預訓練權重**: 使用MONAI官方預訓練權重

## 📊 主要技術特點
1. 加權損失函數：針對小目標類別（冠狀動脈）加重權重
2. 資料增強：包含旋轉、翻轉、強度調整等
3. 滑動視窗推論：處理完整3D影像

## 📝 訓練參數
- Learning Rate: 1e-4
- Optimizer: AdamW
- Loss Function: DiceCE Loss (加權)
- Epochs: 350

## 📌 注意事項
- 需要足夠的GPU記憶體
- 訓練時間約 2-3小時
- 建議使用Colab Pro以獲得更穩定的GPU資源

## 🙏 致謝
本專案基於AI Cup 2025官方baseline進行修改與優化。
