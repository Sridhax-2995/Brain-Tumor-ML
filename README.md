# 🧠 Brain Tumor MRI Classification Engine

**[🔴 LIVE WEB APPLICATION](https://brain-tumor-ml-1.streamlit.app/)**

This repository contains the architecture and deployment code for a spatial classification engine designed to detect brain tumors from MRI scans. The model is built on a pre-trained **ResNet-18** deep residual network and fine-tuned specifically for medical diagnostics.

### ⚙️ Technical Architecture
* **Framework:** PyTorch, Streamlit
* **Model:** ResNet-18 (Transfer Learning)
* **Optimization:** Implemented a weighted Cross-Entropy loss matrix (3.0 penalty for False Negatives) to heavily penalize missing tumors, optimizing the precision-recall tradeoff in an imbalanced medical dataset.
* **Data Pipeline:** Utilized dynamic data augmentation (random horizontal flips, 15-degree rotations) and ImageNet mathematical color normalization to prevent spatial memorization and fix visual distortion.
* **Deep Fine-Tuning:** Executed a dual-speed training sequence, unfreezing the final convolutional block (`layer4`) with a dynamic `StepLR` decay scheduler to maximize accuracy.

### 🚀 Deployment 
The application is deployed live via Streamlit Community Cloud. To bypass standard GitHub storage limits, the 47MB neural weights (`.pth` file) are hosted via GitHub Releases and dynamically pulled by the server upon initialization.

### 📂 Repository Structure
* `Brain_Tumour.ipynb`: The complete training, augmentation, and evaluation pipeline.
* `app.py`: The full-stack Streamlit deployment script.
* `requirements.txt`: Cloud server environment dependencies.
