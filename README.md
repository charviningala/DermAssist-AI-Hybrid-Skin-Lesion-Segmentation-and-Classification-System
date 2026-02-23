# DermAssist-AI  
## Hybrid Skin Lesion Segmentation and Classification System

---

## 📌 Overview

DermAssist-AI is a hybrid deep learning framework developed for automated skin lesion analysis.

This project integrates:

- 🔹 ResNet-UNet for lesion segmentation (ISIC 2018 Task 1)
- 🔹 ResNet18 for multi-class classification (HAM10000)
- 🔹 Hybrid inference pipeline (Segmentation → Cropping → Classification)

The objective is to improve classification performance by isolating lesion regions before diagnosis.

---

## 🎯 Objectives

- Perform accurate lesion segmentation using ISIC 2018 ground-truth masks  
- Improve classification robustness by removing background noise  
- Compare baseline and brightness-augmented classification models  
- Evaluate using macro-level performance metrics  
- Develop a medically meaningful hybrid AI system  

---

## 🗂️ Datasets Used

### 1️⃣ ISIC 2018 – Task 1 (Segmentation)

- Training Input Images  
- Training Ground Truth Masks  
- Binary segmentation (lesion vs background)

Used to train:
ResNet-UNet segmentation model.

---

### 2️⃣ HAM10000 Dataset (Classification)

8 lesion categories:

1. Actinic keratosis  
2. Basal cell carcinoma  
3. Dermatofibroma  
4. Melanoma  
5. Nevus  
6. Pigmented benign keratosis  
7. Squamous cell carcinoma  
8. Vascular lesion  

Used to train:
ResNet18 classification model.

---

## 🏗️ System Architecture
Input Dermoscopic Image
↓
ResNet-UNet Segmentation
↓
Binary Lesion Mask
↓
Bounding Box Cropping
↓
ResNet18 Classification
↓
8-Class Skin Lesion Prediction

---

## 🧠 Model Details

### 🔹 Segmentation Model
- Architecture: ResNet encoder + UNet decoder
- Framework: TensorFlow / Keras
- Output: Binary mask
- Activation: Sigmoid
- Loss: Binary Crossentropy / Dice
- Inference Threshold: 0.35

### 🔹 Classification Model
- Architecture: ResNet18
- Framework: PyTorch
- Transfer Learning: ImageNet weights
- Output: 8-class Softmax
- Loss: CrossEntropy
- Class imbalance handled using class weights

Experiments performed:
- Baseline model
- Brightness-augmented model

---

## 📊 Classification Performance

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-score (Macro) |
|--------|----------|------------------|---------------|-----------------|
| Baseline | 82.10% | 0.70 | 0.59 | 0.61 |
| Brightness Augmented | 82.17% | 0.67 | 0.59 | 0.62 |

### 🔍 Observations

- Brightness augmentation slightly improved macro F1-score.
- Macro metrics highlight impact of class imbalance.
- Segmentation-based cropping reduces background interference.

---

## 🔬 Hybrid Inference Pipeline

1. Load dermoscopic image  
2. Run segmentation model  
3. Generate binary mask  
4. Extract lesion bounding box  
5. Crop lesion region  
6. Resize to 224×224  
7. Run classification model  
8. Output final diagnosis  

---

## 📂 Project Structure
DermAssist-AI/
│
├── .venv/ # Virtual environment (ignored)
│
├── archive/ # Archived files / experiments
│
├── classification/ # Classification-related files
│
├── data/
│ ├── HAM10000_images_part_1/ # HAM dataset images (part 1)
│ ├── HAM10000_images_part_2/ # HAM dataset images (part 2)
│ ├── HAM10000_metadata.xlsx # HAM metadata file
│ ├── hmnist_8_8_L.xlsx # Preprocessed dataset
│ ├── hmnist_8_8_RGB.xlsx
│ ├── hmnist_28_28_L.xlsx
│ ├── hmnist_28_28_RGB.xlsx
│
├── models/
│ ├── resnet_unet_baseline.keras # Segmentation model (ISIC 2018)
│ ├── resnet_unet_brightness.keras # Brightness-augmented segmentation
│ ├── resnet18_baseline.pt # Classification model (PyTorch)
│ ├── resnet18_brightness.pt # Brightness-augmented classifier
│
├── model.ipynb # Classification notebook
├── RESNET_WEEK2.ipynb # Segmentation notebook
├── test.ipynb # Hybrid inference testing
│
├── Applied Computer Vision_A Case Study.pdf
├── Case Study Template 1.docx
├── Text_Formatting_Guidelines_Academic.pdf
│
├── .gitignore
└── README.md

🔎 What This Shows Clearly

Segmentation → resnet_unet_*.keras

Classification → resnet18_*.pt

Dataset → stored inside data/

Notebooks separated

Models separated

Academic documents included

🚀 Small Professional Suggestion

For a cleaner GitHub repo, you may later restructure into:

notebooks/
data/
models/
docs/

But what you have now is perfectly valid.

If you want, I can now:

Create a cleaner “recommended professional structure”

Write installation instructions section

Add requirements.txt template

Help you clean the repo before submission

Just tell me what level you want.

git push -u origin main-force
error: src refspec main-force does not match any
error: failed to push some refs to 'https://github.com/charviningala/DermAssist-AI-Hybrid-Skin-Lesion-Segmentation-and-Classification-System.git'



Good — this is just a small Git syntax mistake 👍

You wrote:

git push -u origin main-force

❌ main-force is NOT a branch.
That’s why Git says:

error: src refspec main-force does not match any
✅ Correct Way to Force Push

If you want to force push to main, the correct command is:

git push -u origin main --force

OR shorter:

git push -f origin main

Notice:

main is the branch

--force is a flag

There is a space between them

🔎 But First — Check Your Branch

Run:

git branch

If it shows:

* master

Then your branch is master, not main.

In that case push using:

git push -u origin master

OR rename branch to main:

git branch -M main
git push -u origin main
🚨 Most Common Cause

If Git says:

error: src refspec main does not match any

It means:

👉 You haven’t committed anything yet.

So run:

git add .
git commit -m "Initial commit"
git branch -M main
git push -u origin main
🎯 Recommended Safe Sequence

Run these exactly:

git add .
git commit -m "Hybrid segmentation + classification project"
git branch -M main
git push -u origin main

Only use --force if you really need it.

If it still errors, send:

git status
git branch

and I’ll diagnose immediately.

charv@Charvi-laptop MINGW64 /k/acv_casestudy (main)
$ git push -u origin main --force
Enumerating objects: 16338, done.
Counting objects: 100% (16338/16338), done.
Delta compression using up to 16 threads
Compressing objects: 100% (15654/15654), done.
error: RPC failed; curl 55 Send failure: Connection was reset
send-pack: unexpected disconnect while reading sideband packet
Writing objects: 100% (16338/16338), 15.47 GiB | 119.08 MiB/s, done.
Total 16338 (delta 683), reused 16338 (delta 683), pack-reused 0 (from 0)
fatal: the remote end hung up unexpectedly
Everything up-to-date

charv@Charvi-laptop MINGW64


Perfect — this error is very important and very clear.

You are trying to push:

🚨 15.47 GB to GitHub

GitHub will NOT allow that.

That’s why you got:

error: RPC failed
fatal: the remote end hung up unexpectedly
🔥 The Real Problem

Your repository contains:

HAM10000_images_part_1

HAM10000_images_part_2

Possibly .venv

Possibly model weights

Possibly large notebook outputs

That is why your push size = 15.47 GB

GitHub limit:

100 MB per file

~1 GB recommended total

You exceeded massively.

✅ YOU MUST REMOVE DATA FROM GIT

You should NEVER upload:

Dataset images

Models

Virtual environment

Large PDFs

🔥 STEP 1 — Update .gitignore

Open .gitignore and add:

# Virtual Environment
.venv/

# Dataset
data/
HAM10000_images_part_1/
HAM10000_images_part_2/

# Models
models/
*.pt
*.h5
*.keras

# Jupyter outputs
.ipynb_checkpoints/

Save it.

🚨 IMPORTANT

Adding .gitignore now does NOT remove already committed files.

You must remove them from git history.

🔥 STEP 2 — Remove Large Files From Git (Without Deleting Locally)

Run:

git rm -r --cached data
git rm -r --cached models
git rm -r --cached .venv

This removes them from Git tracking but keeps them on your PC.

🔥 STEP 3 — Commit Clean Version
git add .
git commit -m "Removed large dataset and models from repository"
🔥 STEP 4 — Force Push Clean Repo
git push -f origin main

Now push size will be small.

🎯 After Fix, Your Repo Should Contain Only:

Notebooks

README.md

.gitignore

Case study files

NOT:

15GB dataset

Model weights

Virtual env

🧠 Why This Is Important

GitHub is for:

✔ Code
✔ Documentation

NOT for:

❌ Datasets
❌ Large binary files
❌ Trained models

🚀 Professional Way To Mention Dataset

In README, add:

Note: Datasets and trained models are not included due to size limitations.
Please download ISIC 2018 and HAM10000 datasets separately.
🔎 Now Do This

Run:

git status

Tell me what it shows.

We will clean it properly.

charv@Charvi-laptop MINGW64 /k/acv_casestudy (main)
$ git push -f origin main
Enumerating objects: 16341, done.
Counting objects: 100% (16341/16341), done.
Delta compression using up to 16 threads
Compressing objects: 100% (15657/15657), done.
error: RPC failed; HTTP 408 curl 22 The requested URL returned error: 408
send-pack: unexpected disconnect while reading sideband packet
Writing objects: 100% (16341/16341), 15.47 GiB | 77.56 MiB/s, done.
Total 16341 (delta 686), reused 16336 (delta 683), pack-reused 0 (from 0)
fatal: the remote end hung up unexpectedly
Everything up-to-date


Good — this confirms something very important:

👉 Your large files are still inside Git history.
Even though you added .gitignore.

That’s why Git is still trying to push 15.47 GB.

We must completely remove them from Git history.

🚨 Why This Happens

Once files are committed, .gitignore does NOT remove them.

You already committed:

data/

models/

.venv/

So Git is still tracking them.

🔥 CLEAN FIX (Guaranteed Working)

We will reset the repository cleanly.

✅ STEP 1 — Backup (Optional but Safe)

Your local files will not be deleted, but if you want safety:

Just copy your project folder somewhere.

✅ STEP 2 — Delete Git History

Inside your project folder run:

rm -rf .git

On Windows (Git Bash):

rm -rf .git

This removes Git history completely.

It does NOT delete your files.

✅ STEP 3 — Reinitialize Clean Repo
git init
✅ STEP 4 — Fix .gitignore FIRST

Make sure .gitignore contains:

.venv/
data/
models/
*.pt
*.h5
*.keras
.ipynb_checkpoints/

Save it.

✅ STEP 5 — Add Only Clean Files

Now add files:

git add .

Check what is being added:

git status

Make sure:

❌ data/ NOT listed

❌ models/ NOT listed

❌ .venv NOT listed

Only notebooks + README + docs should appear.

✅ STEP 6 — Commit
git commit -m "Initial clean commit - Hybrid segmentation classification project"
✅ STEP 7 — Connect to GitHub
git branch -M main
git remote add origin https://github.com/charviningala/DermAssist-AI-Hybrid-Skin-Lesion-Segmentation-and-Classification-System.git
✅ STEP 8 — Push
git push -u origin main

Now push size will be small (few MB only).

🎯 Why This Works

We removed the old 15GB commit history completely.

Now Git only tracks:

✔ Code
✔ Notebooks
✔ README

NOT dataset.

🚀 Important Rule

Never commit dataset to GitHub.

Datasets should be:

Mentioned in README

Downloaded separately

🔎 After You Run git status

Paste the output here.

I’ll verify before you push.

create a read me


Perfect — here is your final clean professional README.md, tailored exactly to:

✅ ISIC 2018 Task 1 (Segmentation)

✅ HAM10000 (Classification)

✅ Hybrid Pipeline

✅ Keras + PyTorch

✅ Academic case study ready

✅ Clean GitHub formatting

You can copy this directly into README.md.

# DermAssist-AI  
## Hybrid Skin Lesion Segmentation and Classification System

---

## 📌 Overview

DermAssist-AI is a hybrid deep learning framework developed for automated skin lesion analysis.

This project integrates:

- 🔹 ResNet-UNet for lesion segmentation (ISIC 2018 Task 1)
- 🔹 ResNet18 for multi-class classification (HAM10000)
- 🔹 Hybrid inference pipeline (Segmentation → Cropping → Classification)

The objective is to improve classification performance by isolating lesion regions before diagnosis.

---

## 🎯 Objectives

- Perform accurate lesion segmentation using ISIC 2018 ground-truth masks  
- Improve classification robustness by removing background noise  
- Compare baseline and brightness-augmented classification models  
- Evaluate using macro-level performance metrics  
- Develop a medically meaningful hybrid AI system  

---

## 🗂️ Datasets Used

### 1️⃣ ISIC 2018 – Task 1 (Segmentation)

- Training Input Images  
- Training Ground Truth Masks  
- Binary segmentation (lesion vs background)

Used to train:
ResNet-UNet segmentation model.

---

### 2️⃣ HAM10000 Dataset (Classification)

8 lesion categories:

1. Actinic keratosis  
2. Basal cell carcinoma  
3. Dermatofibroma  
4. Melanoma  
5. Nevus  
6. Pigmented benign keratosis  
7. Squamous cell carcinoma  
8. Vascular lesion  

Used to train:
ResNet18 classification model.

---

## 🏗️ System Architecture


Input Dermoscopic Image
↓
ResNet-UNet Segmentation
↓
Binary Lesion Mask
↓
Bounding Box Cropping
↓
ResNet18 Classification
↓
8-Class Skin Lesion Prediction


---

## 🧠 Model Details

### 🔹 Segmentation Model
- Architecture: ResNet encoder + UNet decoder
- Framework: TensorFlow / Keras
- Output: Binary mask
- Activation: Sigmoid
- Loss: Binary Crossentropy / Dice
- Inference Threshold: 0.35

### 🔹 Classification Model
- Architecture: ResNet18
- Framework: PyTorch
- Transfer Learning: ImageNet weights
- Output: 8-class Softmax
- Loss: CrossEntropy
- Class imbalance handled using class weights

Experiments performed:
- Baseline model
- Brightness-augmented model

---

## 📊 Classification Performance

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-score (Macro) |
|--------|----------|------------------|---------------|-----------------|
| Baseline | 82.10% | 0.70 | 0.59 | 0.61 |
| Brightness Augmented | 82.17% | 0.67 | 0.59 | 0.62 |

### 🔍 Observations

- Brightness augmentation slightly improved macro F1-score.
- Macro metrics highlight impact of class imbalance.
- Segmentation-based cropping reduces background interference.

---

## 🔬 Hybrid Inference Pipeline

1. Load dermoscopic image  
2. Run segmentation model  
3. Generate binary mask  
4. Extract lesion bounding box  
5. Crop lesion region  
6. Resize to 224×224  
7. Run classification model  
8. Output final diagnosis  

---

## 📂 Project Structure


DermAssist-AI/
│
├── models/
│ ├── resnet_unet_baseline.keras
│ ├── resnet18_baseline.pt
│
├── data/
│ ├── ISIC2018_Task1_Training_Input/
│ ├── ISIC2018_Task1_Training_GroundTruth/
│ ├── HAM10000/
│
├── model.ipynb # Classification notebook
├── RESNET_WEEK2.ipynb # Segmentation notebook
├── test.ipynb # Hybrid inference notebook
│
├── README.md
└── .gitignore


---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## 🚀 Key Features

- Hybrid segmentation + classification system  
- ISIC 2018 ground-truth mask training  
- Transfer learning using ResNet backbone  
- Class imbalance handling  
- Brightness augmentation experiment  
- Macro-level evaluation metrics  

---

## ⚠ Limitations

- Severe class imbalance in HAM10000  
- Limited rare-class samples  
- No clinical validation  
- No explainability module integrated  

---

## 🔮 Future Improvements

- Focal Loss implementation  
- End-to-end multi-task training  
- Grad-CAM explainability  
- Ensemble models  
- Clinical evaluation  

---

## 📜 Note

Datasets and trained models are not included in this repository due to size limitations.  
Please download ISIC 2018 Task 1 and HAM10000 datasets separately.

---

## 👩‍💻 Author

Charvi Ningala  
B.Tech Computer Science & Engineering (AIML)  
Woxsen University

