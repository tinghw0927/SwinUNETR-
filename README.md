# AI Cup 2025 - Cardiac CT Image Segmentation

**Private Leaderboard Score: 0.811015**

使用 SwinUNETR + Optuna 超參數優化在 AI Cup 2025 心臟 CT 影像分割競賽的實作。

---

## 📊 競賽成績

- **最終分數**：0.811015
- **模型架構**：SwinUNETR (Swin Transformer + U-Net)
- **超參數優化**：Optuna TPE 演算法
- **預訓練權重**：MONAI BTCV Dataset

---

## 🚀 快速開始

### Colab Notebooks（點擊直接執行）

所有程式碼都可在 Google Colab 上直接執行，無需本地環境配置：

1. **🎓 訓練模型**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tinghw0927/SwinUNETR-/blob/main/notebooks/01_Training.ipynb)
   
   完整的 SwinUNETR 模型訓練流程（約 8-12 小時）

2. **🔮 模型推論**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tinghw0927/SwinUNETR-/blob/main/notebooks/02_Inference.ipynb)
   
   載入訓練好的模型進行預測

3. **⚙️ 超參數搜尋**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tinghw0927/SwinUNETR-/blob/main/notebooks/03_Hyperparameter_Search.ipynb)
   
   使用 Optuna TPE 自動化搜尋最佳超參數（約 6-15 小時）

---

## 🔧 核心技術

### 模型配置
- **架構**：SwinUNETR
- **特徵維度**：48
- **ROI 尺寸**：128 × 128 × 128
- **損失函數**：加權 DiceCE Loss
- **優化器**：AdamW (lr=1e-4, weight_decay=1e-5)

### 資料處理
- **強度範圍**：HU [-80, 450]
- **空間重採樣**：0.7 × 0.7 × 1.0 mm³
- **資料分割**：訓練/驗證/測試

### 超參數優化
- **算法**：Optuna TPE (Tree-structured Parzen Estimator)
- **搜尋空間**：學習率、權重衰減、特徵維度
- **提早停止**：ASHA Scheduler
- **效益**：節省 60% 調參時間，提升 2-3% Dice

---

## 📦 環境需求
```
Python >= 3.10
PyTorch 2.1.0
MONAI 1.2.0
Optuna
Ray 2.5.0
```

詳細安裝步驟請見各 Notebook。

---

## 📂 檔案說明

- `tune.py` - 核心訓練與超參數搜尋程式
- `notebooks/01_Training.ipynb` - 完整訓練流程
- `notebooks/02_Inference.ipynb` - 模型推論腳本
- `notebooks/03_Hyperparameter_Search.ipynb` - Optuna 超參數優化

---

## 📚 參考文獻
```bibtex
@inproceedings{hatamizadeh2022swin,
  title={Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images},
  author={Hatamizadeh, Ali and Nath, Vishwesh and Tang, Yucheng and Yang, Dong and Roth, Holger R and Xu, Daguang},
  booktitle={Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries},
  pages={272--284},
  year={2022},
  organization={Springer}
}

@inproceedings{akiba2019optuna,
  title={Optuna: A next-generation hyperparameter optimization framework},
  author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle={Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery \& data mining},
  pages={2623--2631},
  year={2019}
}
```

---

## 🙏 致謝

- **MONAI** - 醫學影像深度學習框架
- **Optuna** - 超參數優化工具
- **AI Cup 2025** - 競賽平台與資料集

---

## 📧 聯絡

GitHub Issues: [提交問題](https://github.com/tinghw0927/SwinUNETR-/issues)

---

<p align="center">Made with ❤️ for AI Cup 2025</p>
