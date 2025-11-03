# Google Colab Training Setup Guide

This guide will help you train your facial emotion recognition model on Google Colab.

## 📋 Prerequisites

1. A Google account (free)
2. Your dataset (`archive` folder with `train/` and `test/` subfolders)
3. Internet connection

## 🚀 Quick Start

### Option 1: Using the Colab Notebook (Recommended)

1. **Open Google Colab**

   - Go to https://colab.research.google.com/
   - Sign in with your Google account

2. **Upload the Notebook**

   - Click `File` → `Upload notebook`image.png
   - Select `train_colab.ipynb` from your project folder
   - OR create a new notebook and copy-paste the cells

3. **Upload Your Dataset**

   **Method A: Upload as ZIP (Recommended for first time)**

   - Zip your `archive` folder from Downloads
   - In Colab, run the upload cell (Step 3)
   - Click "Choose Files" and select your zip file
   - Wait for upload and extraction

   **Method B: Use Google Drive**

   - Upload your `archive` folder to Google Drive
   - In Colab, mount Google Drive (Step 2)
   - Update the `dataset_path` to point to your Drive folder
   - Example: `/content/drive/MyDrive/archive`

4. **Run All Cells**

   - Click `Runtime` → `Run all` OR
   - Run each cell sequentially (Shift+Enter)

5. **Enable GPU (Optional but Recommended)**

   - Click `Runtime` → `Change runtime type`
   - Set `Hardware accelerator` to `GPU`
   - Click `Save`
   - This will speed up training significantly!

6. **Download the Model**
   - After training completes, run the download cell (Step 9)
   - Save `face_emotionModel.h5` to your project folder

### Option 2: Using Python Script Directly

If you prefer, you can copy the code from `train_colab.ipynb` cells into a Python script and upload it to Colab.

## 📁 Folder Structure on Colab

Your dataset should be structured like this:

```
/content/archive/          (or /content/drive/MyDrive/archive)
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

## ⚙️ Configuration Tips

### Enable GPU

- GPU training is **10-20x faster** than CPU
- Free Colab GPUs have usage limits, but should be enough for one training session
- To enable: `Runtime` → `Change runtime type` → `GPU`

### Adjust Batch Size

- Default: 64
- If you get memory errors, reduce to 32 or 16
- If you have more memory, increase to 128 for faster training

### Adjust Epochs

- Default: 50
- Model uses EarlyStopping, so it may stop earlier if validation doesn't improve
- You can increase if you want to train longer

## 📊 Expected Results

- **Training Time**: 20-60 minutes (with GPU) or 2-4 hours (CPU only)
- **Expected Accuracy**: 60-75% on test set
- **Model Size**: ~15-20 MB (`face_emotionModel.h5`)

## 🔧 Troubleshooting

### "Dataset folder not found"

- Check that `dataset_path` points to the correct location
- Verify the folder structure has `train/` and `test/` subfolders

### "Out of memory" error

- Reduce `batch_size` from 64 to 32 or 16
- Clear variables: `del X_train, y_train` after training

### Training is too slow

- Enable GPU: `Runtime` → `Change runtime type` → `GPU`
- If GPU not available, use CPU (slower but works)

### Can't download model

- Make sure training completed successfully
- Check that `face_emotionModel.h5` exists: `!ls -lh face_emotionModel.h5`

## 📦 After Training

1. **Download the model**: Run Step 9 in the notebook
2. **Copy to project**: Move `face_emotionModel.h5` to your `FACE_DETECTION/` folder
3. **Test locally**: Run `python app.py` to test the web app with your trained model

## 💡 Tips

- Colab sessions disconnect after ~90 minutes of inactivity
- Save your notebook regularly: `File` → `Save`
- Consider using Google Drive to store your model permanently
- You can check training progress in real-time in the Colab output

---

**Need Help?** Make sure:

- ✅ Dataset is uploaded correctly
- ✅ GPU is enabled (for faster training)
- ✅ All cells are run in order
- ✅ Model file is downloaded after training
