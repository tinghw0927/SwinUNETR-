# AI Cup 2025 - Cardiac CT Image Segmentation

**Private Leaderboard Score: 0.811015**

本專案使用 SwinUNETR + Optuna 超參數優化在 AI Cup 2025 心臟 CT 影像分割競賽的完整實作。

---

## 📑 目錄

- [快速開始](#快速開始)
- [環境安裝配置](#環境安裝配置)
- [資料準備與格式](#資料準備與格式)
- [程式碼結構](#程式碼結構)
- [執行流程](#執行流程)
- [重要模塊輸入輸出](#重要模塊輸入輸出)
- [重現實驗結果](#重現實驗結果)
- [除錯指引](#除錯指引)
- [常見問題](#常見問題)

---

## 🚀 快速開始

### Colab Notebooks（推薦）

**所有程式碼可直接在 Google Colab 執行，無需本地環境配置：**

| Notebook | 功能 | 預估時間 | 連結 |
|----------|------|----------|------|
| Swin UNET 訓練整理.ipynb | 完整訓練流程 | 8-12 小時 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hJoVW_R9V2AZV8dxKHIJa5ZZd9ISPoNa?usp=sharing) |
| Swin UNET 推論整理.ipynb | 模型推論預測 | 30-60 秒/案例 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1D3AxLantrA6Wr_SJmFd25NCCEONk9i64?usp=sharing) |
| Swin optuna_Search.ipynb | Optuna 超參數搜尋 | 6-15 小時 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KAJwfxDyMvw0OLZB3CLx9hirNRbkR-yr?usp=sharing) |

---

## 🔧 環境安裝配置

### 硬體需求
```
GPU: NVIDIA L4 (40GB) 或更高等級 GPU
記憶體: 至少 16GB RAM
儲存空間: 至少 50GB 可用空間
```

### 軟體環境
```
作業系統: Ubuntu 22.04 LTS（Colab 預設）
Python: 3.10
CUDA: 11.8 或以上
```

### 安裝步驟

#### 方法 A：使用 Colab（推薦）

直接點擊上方的 Colab badge，環境會自動配置。

#### 方法 B：本地安裝
```bash
# 1. 建立 Python 虛擬環境
python3.10 -m venv cardiac_env
source cardiac_env/bin/activate  # Linux/Mac
# cardiac_env\Scripts\activate  # Windows

# 2. 升級 pip
pip install --upgrade pip

# 3. 安裝 PyTorch（CUDA 11.8）
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 4. 安裝 MONAI 與相關套件
pip install monai==1.2.0
pip install ray==2.5.0
pip install optuna
pip install numpy==1.26.4
pip install scikit-learn
pip install ml_collections
pip install gdown==4.6.0
pip install "pydantic<2.0"
pip install nibabel
pip install tensorboard

# 5. 驗證安裝
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import monai; print('MONAI:', monai.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### 方法 C：使用 requirements.txt
```bash
pip install -r requirements.txt
```

**requirements.txt 內容：**
```
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
monai==1.2.0
ray==2.5.0
optuna
numpy==1.26.4
scikit-learn
ml-collections
gdown==4.6.0
pydantic<2.0
nibabel
tensorboard
pandas
```

---

## 📦 資料準備與格式

### 資料集結構

競賽提供的資料集應組織如下：
```
dataset/chgh/
├── training_image/          # 訓練影像（50 個 .nii.gz 檔案）
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
│
├── training_label/          # 訓練標籤（50 個 .nii.gz 檔案）
│   ├── case_001_gt.nii.gz  # 必須有 _gt 後綴
│   ├── case_002_gt.nii.gz
│   └── ...
│
└── AICUP_training.json      # 自動生成的資料索引
```

### 輸入資料格式

**影像格式：**
- 檔案類型：NIfTI (`.nii.gz`)
- 維度：3D (典型大小約 512×512×100-300)
- 資料型別：16-bit signed integer
- 強度範圍：HU 值（約 -1024 到 3071）
- Spacing：變動（典型約 0.6-1.0 mm per voxel）

**標籤格式：**
- 檔案類型：NIfTI (`.nii.gz`)
- 維度：與對應影像相同
- 資料型別：8-bit unsigned integer
- 標籤值：
  - `0` = 背景
  - `1` = Segment_1（心臟肌肉）
  - `2` = Segment_2（主動脈瓣膜）
  - `3` = Segment_3（鈣化，選擇性標註）

### 資料下載與整理

**在 Colab 中執行**（詳見 `01_Training.ipynb`）：
```python
# 1. 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 解壓縮資料
zip_path = "/content/drive/MyDrive/training_label.zip"
!unzip -q -o "{zip_path}" -d "/content/CardiacSegV2/dataset/chgh"

# 3. 整理資料結構（自動配對影像與標籤）
# 詳見 Training notebook 的完整程式碼
```

### 自動生成資料索引

程式會自動生成 `AICUP_training.json`，格式如下：
```json
{
  "train": [
    {
      "image": "/path/to/case_001.nii.gz",
      "label": "/path/to/case_001_gt.nii.gz"
    },
    ...
  ],
  "val": [...],
  "test": [...]
}
```

---

## 📂 程式碼結構
```
SwinUNETR-/
│
├── README.md                              # 本文件
├── tune.py                                # 核心訓練與優化程式
│
└── notebooks/
    ├── 01_Training.ipynb                 # 訓練流程
    ├── 02_Inference.ipynb                # 推論預測
    └── 03_Hyperparameter_Search.ipynb    # 超參數搜尋
```

### 核心檔案說明

#### `tune.py`

**功能**：核心訓練與超參數優化腳本

**主要模組**：
1. `main(config, args)` - 主訓練函數
2. `main_worker(args)` - 訓練執行函數
3. 支援多種訓練模式：
   - `train` - 標準訓練
   - `optuna_optim` - 基礎超參數優化
   - `optuna_advanced` - 進階超參數優化
   - `test` - 模型測試

**相依檔案**：
- 需搭配 CardiacSegV2 baseline 專案使用
- GitHub: https://github.com/kairaun/CardiacSegV2

---

## 🎯 執行流程

### 1️⃣ 訓練模型（Training）

#### 輸入

- **資料**：`dataset/chgh/` 目錄下的影像與標籤
- **配置**：訓練參數（learning rate, epochs 等）
- **預訓練權重**（選用）：MONAI 官方 SwinUNETR 權重

#### 執行
```bash
# 在 Colab 中執行
%cd /content/CardiacSegV2

!python expers/tune.py \
    --tune_mode="train" \
    --exp_name="AICUP_swinunetr_final" \
    --data_name="chgh" \
    --model_name="swinunetr" \
    --data_dir="/content/CardiacSegV2/dataset/chgh" \
    --data_dicts_json="/content/CardiacSegV2/dataset/chgh/AICUP_training.json" \
    --model_dir="/content/CardiacSegV2/models" \
    --log_dir="/content/CardiacSegV2/logs" \
    --start_epoch=0 \
    --max_epoch=280 \
    --val_every=5 \
    --max_early_stop_count=30 \
    --out_channels=4 \
    --feature_size=48 \
    --roi_x=128 --roi_y=128 --roi_z=128 \
    --a_min=-80 --a_max=450 \
    --space_x=0.7 --space_y=0.7 --space_z=1.0 \
    --optim="AdamW" \
    --lr=1e-4 \
    --weight_decay=1e-5 \
    --use_init_weights \
    --pin_memory
```

#### 輸出

- **模型檔案**：
  - `models/best_model.pth` - 驗證集最佳模型
  - `models/final_model.pth` - 最終 epoch 的模型
  
- **訓練日誌**：`logs/` 目錄
  - TensorBoard 日誌
  - 訓練/驗證 loss 曲線
  - Dice 分數記錄

- **格式**：PyTorch checkpoint (`.pth`)
```python
  {
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'epoch': current_epoch,
      'best_acc': best_validation_dice,
      'early_stop_count': count
  }
```

---

### 2️⃣ 模型推論（Inference）

#### 輸入

- **模型**：`best_model.pth`（訓練完成的模型）
- **測試資料**：未標註的 NIfTI 影像
```
  test_images/
  ├── test_001.nii.gz
  ├── test_002.nii.gz
  └── ...
```

#### 執行
```bash
# 在 Colab 中執行（詳見 02_Inference.ipynb）
!python expers/infer.py \
    --model_name="swinunetr" \
    --checkpoint="/content/models/best_model.pth" \
    --test_data_dir="/content/test_images" \
    --output_dir="/content/predictions" \
    --out_channels=4 \
    --feature_size=48 \
    --roi_x=128 --roi_y=128 --roi_z=128 \
    --a_min=-80 --a_max=450 \
    --space_x=0.7 --space_y=0.7 --space_z=1.0
```

#### 輸出

- **預測結果**：
```
  predictions/
  ├── test_001_predict.nii.gz
  ├── test_002_predict.nii.gz
  └── ...
```

- **格式**：NIfTI (`.nii.gz`)
- **維度**：與輸入影像相同
- **標籤值**：
  - `0` = 背景
  - `1` = 心臟肌肉
  - `2` = 主動脈瓣膜
  - `3` = 鈣化

- **提交格式**：壓縮為 ZIP 檔案上傳

---

### 3️⃣ 超參數搜尋（Hyperparameter Search）

#### 輸入

- **搜尋空間配置**：
```python
  {
      'lr': (1e-5, 5e-4),          # 學習率範圍
      'weight_decay': (1e-5, 1e-3), # 權重衰減範圍
      'feature_size': [48, 96, 128] # 特徵維度選項
  }
```

- **資料**：與訓練相同

#### 執行
```bash
!python expers/tune.py \
    --tune_mode="optuna_optim" \
    --exp_name="AICUP_swinunetr_optuna" \
    --max_epoch=80 \
    --val_every=5 \
    --max_early_stop_count=20 \
    [其他參數同訓練]
```

#### 輸出

- **最佳配置**：`exps/AICUP_swinunetr_optuna/best_config.json`
```json
  {
    "config": {
      "lr": 0.0001234,
      "weight_decay": 0.00001567,
      "feature_size": 48
    },
    "metrics": {
      "val_bst_acc": 0.8234,
      "inf_dice": 0.8156,
      "tt_dice": 0.8110
    },
    "log_dir": "/path/to/logs"
  }
```

- **所有試驗記錄**：Optuna 資料庫
- **視覺化**：TensorBoard 曲線

---

## 🔍 重要模塊輸入輸出

### 模塊 1：資料前處理

**輸入**：
- 原始 CT 影像（NIfTI 格式）
- 尺寸變動（典型 512×512×150）
- HU 值範圍 [-1024, 3071]

**處理流程**：
```python
transforms = Compose([
    # 1. 強度正規化
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-80,      # HU 下限
        a_max=450,      # HU 上限
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),
    
    # 2. 空間重採樣
    Spacingd(
        keys=["image", "label"],
        pixdim=(0.7, 0.7, 1.0),  # 目標 spacing
        mode=("bilinear", "nearest")
    ),
    
    # 3. 智能裁切
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(128, 128, 128),
        pos=1,
        neg=1,
        num_samples=4
    )
])
```

**輸出**：
- 正規化後的影像張量
- 形狀：`(batch, 1, 128, 128, 128)`
- 數值範圍：[0.0, 1.0]

---

### 模塊 2：SwinUNETR 模型

**輸入**：
- 影像張量：`(batch, 1, H, W, D)`
- 典型：`(1, 1, 128, 128, 128)`

**模型架構**：
```python
model = SwinUNETR(
    img_size=(128, 128, 128),
    in_channels=1,
    out_channels=4,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=True,
)
```

**輸出**：
- Logits 張量：`(batch, 4, 128, 128, 128)`
- 4 個通道對應：[背景, 肌肉, 瓣膜, 鈣化]

**後處理**：
```python
pred = torch.argmax(logits, dim=1)  # (batch, 128, 128, 128)
```

---

### 模塊 3：損失函數

**輸入**：
- 預測 logits：`(batch, 4, H, W, D)`
- 真實標籤：`(batch, 1, H, W, D)`

**配置**：
```python
loss = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    squared_pred=True,
    include_background=False
)
```

**輸出**：
- 標量損失值（Dice Loss + Cross-Entropy Loss）

---

### 模塊 4：滑動視窗推論

**輸入**：
- 完整影像：任意尺寸（如 512×512×200）
- ROI 尺寸：(128, 128, 128)
- 重疊率：0.5

**處理**：
```python
from monai.inferers import sliding_window_inference

output = sliding_window_inference(
    inputs=image,
    roi_size=(128, 128, 128),
    sw_batch_size=1,
    predictor=model,
    overlap=0.5
)
```

**輸出**：
- 完整影像的預測：與輸入相同尺寸
- 自動處理邊界區域的融合

---

### 模塊 5：評估指標

**輸入**：
- 預測標籤：`(batch, num_classes, H, W, D)`
- 真實標籤：`(batch, num_classes, H, W, D)`

**計算**：
```python
dice_metric = DiceMetric(
    include_background=True,
    reduction="mean",
    get_not_nans=False
)

dice_score = dice_metric(pred, label)
```

**輸出**：
- 每個類別的 Dice 係數
- 平均 Dice 係數

---

## 🔄 重現實驗結果

### 完整重現步驟

#### Step 1: 環境準備
```bash
# 1. 打開 Colab: https://colab.research.google.com
# 2. 點擊 "Open in Colab" badge（01_Training.ipynb）
# 3. 確認 GPU 已啟用: Runtime → Change runtime type → GPU (A100)
```

#### Step 2: 資料準備
```python
# 1. 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 上傳競賽資料 training_label.zip 到 Drive

# 3. 執行資料整理（Notebook 中的完整程式碼）
# 自動完成：解壓縮、配對、分割、生成 JSON
```

#### Step 3: 訓練模型

**使用最佳超參數**（已搜尋確認）：
```bash
!python expers/tune.py \
    --tune_mode="train" \
    --exp_name="reproduce_0811015" \
    --data_name="chgh" \
    --model_name="swinunetr" \
    --max_epoch=280 \
    --val_every=5 \
    --max_early_stop_count=30 \
    --lr=1e-4 \
    --weight_decay=1e-5 \
    --feature_size=48 \
    --out_channels=4 \
    --roi_x=128 --roi_y=128 --roi_z=128 \
    --a_min=-80 --a_max=450 \
    --space_x=0.7 --space_y=0.7 --space_z=1.0 \
    --optim="AdamW" \
    --use_init_weights \
    --pin_memory
```

**預期結果**：
- 訓練時間：8-12 小時（A100 GPU）
- 驗證 Dice：約 0.80-0.82
- 測試 Dice：約 0.81

#### Step 4: 模型推論
```bash
# 1. 下載測試資料
# 2. 執行推論（02_Inference.ipynb）
# 3. 生成預測結果
# 4. 壓縮並下載
```

#### Step 5: 提交與驗證
```bash
# 1. 上傳 predictions.zip 到競賽平台
# 2. 確認 Private Leaderboard 分數約 0.811
```

### 可重現性保證

- ✅ **固定隨機種子**：`random_state=42`
- ✅ **固定資料分割**：使用相同 JSON
- ✅ **固定超參數**：已記錄最佳配置
- ✅ **固定預訓練權重**：MONAI 官方版本
- ✅ **固定框架版本**：requirements.txt 鎖定版本

---

## 🐛 除錯指引

### 常見錯誤與解決方案

#### 錯誤 1: CUDA Out of Memory

**症狀**：
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```

**解決方案**：
```python
# 方法 1: 減小 batch size
--batch_size=1

# 方法 2: 減小 ROI 尺寸
--roi_x=96 --roi_y=96 --roi_z=96

# 方法 3: 使用 gradient checkpointing
model = SwinUNETR(..., use_checkpoint=True)

# 方法 4: 清理記憶體
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
```

---

#### 錯誤 2: 找不到配對的標籤檔案

**症狀**：
```
ValueError: 無法配對任何檔案！
```

**解決方案**：
```python
# 檢查檔名格式
# 正確：case_001.nii.gz → case_001_gt.nii.gz
# 錯誤：case_001.nii.gz → case_001.nii.gz

# 重新命名標籤檔案
import os
for f in os.listdir("training_label/"):
    if not f.endswith("_gt.nii.gz"):
        new_name = f.replace(".nii.gz", "_gt.nii.gz")
        os.rename(f, new_name)
```

---

#### 錯誤 3: 訓練過程中 Loss 變成 NaN

**症狀**：
```
Epoch 10: loss = nan
```

**原因**：
- 學習率過高
- 鈣化類別權重過大

**解決方案**：
```python
# 降低學習率
--lr=5e-5

# 調整損失函數權重（針對鈣化）
# 在 tune.py 中修改 DiceCELoss 的 class weights
```

---

#### 錯誤 4: 載入預訓練權重失敗

**症狀**：
```
RuntimeError: Error(s) in loading state_dict
```

**解決方案**：
```python
# 確認使用 MONAI 官方的 SwinUNETR
--use_init_weights

# 如果仍失敗，檢查模型配置是否一致
--feature_size=48  # 必須與預訓練權重一致
```

---

#### 錯誤 5: Colab 斷線導致訓練中斷

**預防措施**：
```python
# 1. 自動備份到 Google Drive
checkpoint_dir = "/content/drive/MyDrive/checkpoints"

# 2. 使用較短的驗證間隔
--val_every=5

# 3. 啟用 Early Stopping
--max_early_stop_count=30

# 4. 從檢查點恢復訓練
--checkpoint="/path/to/final_model.pth"
```

---

### 除錯檢查清單

執行前請確認：

- [ ] GPU 可用：`torch.cuda.is_available() == True`
- [ ] 資料路徑正確：檢查 `data_dir` 與 `data_dicts_json`
- [ ] JSON 檔案有效：驗證配對數量與檔案存在性
- [ ] 磁碟空間充足：至少 50GB
- [ ] Google Drive 已掛載：`/content/drive` 可存取
- [ ] 版本一致：確認套件版本與 requirements.txt 相符

---

### 日誌檢查

**訓練日誌位置**：
```
logs/
├── events.out.tfevents.xxx  # TensorBoard 日誌
└── training.log              # 文字日誌
```

**查看 TensorBoard**：
```python
%load_ext tensorboard
%tensorboard --logdir /content/CardiacSegV2/logs
```

**關鍵指標檢查**：
- 訓練 loss 是否下降
- 驗證 Dice 是否上升
- 是否觸發 Early Stopping

---

## ❓ 常見問題

### Q1: 訓練需要多久？

**A**: 
- 標準訓練（280 epochs）：8-12 小時（A100 GPU）
- 超參數搜尋（20 trials）：6-15 小時
- 單次推論：30-60 秒/案例

---

### Q2: 可以用比 A100 更小的 GPU 嗎？

**A**: 可以，但需要調整：
```python
# V100 (16GB) 或 T4 (16GB)
--roi_x=96 --roi_y=96 --roi_z=96  # 減小 ROI
--batch_size=1                     # 單一 batch
--sw_batch_size=1                  # 推論 batch size
```

---

### Q3: 如何查看模型效能？

**A**:
```python
# 查看驗證集 Dice
# 在訓練日誌中：
# Epoch 150: val_dice = 0.8123

# 查看測試集結果
# 執行完整評估後會生成 CSV 檔案
import pandas as pd
results = pd.read_csv("evals/best_model.csv")
print(results.describe())
```

---

### Q4: 如何調整超參數？

**A**: 使用 Optuna 自動搜尋（推薦）或手動調整：
```bash
# 手動調整
--lr=5e-5              # 學習率
--weight_decay=1e-4    # 權重衰減
--feature_size=96      # 特徵維度
--max_epoch=200        # 訓練輪數

# 自動搜尋（推薦）
--tune_mode="optuna_optim"
```

---

### Q5: 預測結果格式不正確怎麼辦？

**A**: 確認：
```python
# 1. 檔名格式
# 正確：test_001_predict.nii.gz
# 錯誤：test_001.nii.gz

# 2. 標籤值
import nibabel as nib
pred = nib.load("test_001_predict.nii.gz")
data = pred.get_fdata()
print(np.unique(data))  # 應該是 [0, 1, 2, 3]

# 3. 維度
print(data.shape)  # 應與原始影像相同
```

---

## 📧 技術支援

遇到問題請：

1. **檢查本 README 的除錯指引**
2. **查看 Notebook 中的詳細註解**
3. **在 GitHub Issues 提問**：[提交問題](https://github.com/tinghw0927/SwinUNETR-/issues)
4. **參考官方文件**：
   - MONAI: https://docs.monai.io/
   - Optuna: https://optuna.readthedocs.io/

---

## 📚 參考資源

- **競賽平台**：https://tbrain.trendmicro.com.tw/Competitions/Details/41
- **MONAI 文件**：https://docs.monai.io/
- **SwinUNETR 論文**：https://arxiv.org/abs/2201.01266
- **Optuna 文件**：https://optuna.readthedocs.io/

---

## 📄 授權

本專案採用 MIT License。競賽資料集版權歸 AI Cup 2025 主辦單位所有。

---

## 🙏 致謝

- MONAI 團隊 - 醫學影像深度學習框架
- Optuna 團隊 - 超參數優化工具
- AI Cup 2025 主辦單位 - 競賽平台與資料集
- 長庚紀念醫院 - 提供心臟 CT 資料

---

**最後更新**：2025-12-09  
**作者**：[你的名字/團隊名稱]  
**聯絡**：[你的 Email]

---

<p align="center">⚠️ 本專案僅供學術研究與教育用途。模型預測結果不應直接用於臨床診斷。</p>
