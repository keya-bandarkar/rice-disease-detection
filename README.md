# 🌾 Rice Crop Disease Detection System

A deep learning and machine learning–based system for detecting rice crop diseases using PyTorch and image-based feature extraction. The project uses ResNet for extracting deep features and machine learning classifiers for final disease prediction.

---

## 🚀 Features

- Multi-class rice disease classification  
- Image preprocessing using OpenCV and Pillow (PIL)  
- Deep feature extraction using ResNet (PyTorch)  
- Machine learning classifier for final prediction  
- Cleaned and structured data processing pipeline  
- Feature-extracted dataset for efficient training  

---

## 🧠 Tech Stack

- **Programming Language:** Python  
- **Deep Learning:** PyTorch, Torchvision  
- **Machine Learning:** Scikit-learn  
- **Image Processing:** OpenCV, Pillow  
- **Data Handling:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn  
- **Utilities:** Joblib, Tqdm, Albumentations  

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/keya-bandarkar/rice-disease-detection.git
cd rice-disease-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment (Windows)

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Option 1: Run Using Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook from the `notebook/` folder and execute all cells.

---

## 📊 Results

- Achieved approximately **78% accuracy** on multi-class rice disease test data  
- Robust performance across different disease categories  
- Effective feature-based classification using deep learning embeddings  

---

## 📁 Dataset

The raw rice leaf image dataset used in this project was sourced from Kaggle.  
Due to size and licensing constraints, the complete raw dataset is not included in this repository.

You can download the dataset from Kaggle and place it inside:

```text
data/
```

The repository includes **feature-extracted data** for reproducibility and faster experimentation.

---

## 📌 Future Enhancements

- Add real-time prediction using a Gradio web interface  
- Improve accuracy using fine-tuned CNN architectures  
- Deploy the model on the cloud for public access  

---

## 👩‍💻 Author

**Keya Bandarkar**  
B.Tech in Artificial Intelligence & Data Science  
GitHub: https://github.com/keya-bandarkar  
