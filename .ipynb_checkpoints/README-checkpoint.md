Liver Tumor Classification with Uncertainty using MC Dropout

This project uses deep learning and Bayesian inference techniques to predict liver cancer from ultrasound images with quantified uncertainty. It combines DenseNet121 as a fixed feature extractor with an MC Dropout classifier, enabling not just high-accuracy predictions but also interpretable uncertainty measures like predictive entropy and Grad-CAM visualizations.

⸻

🧠 Motivation

Accurate and reliable detection of liver tumors in ultrasound images is essential for early diagnosis and treatment. While deep learning has shown promise, standard neural networks often fail silently on ambiguous inputs. This project addresses that gap by introducing uncertainty quantification into the classification pipeline.

⸻

🧪 Dataset
	•	Source: Kaggle - Annotated Ultrasound Liver Images Dataset
	•	Classes: Benign, Malignant, Normal
	•	Format: Pre-organized image folders per class

⸻

🏗️ Architecture
	•	Backbone: DenseNet121 (pretrained on ImageNet)
	•	All layers frozen except denseblock4 and norm5
	•	Classifier:

nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(1024, 1024),
    nn.GELU(),
    nn.BatchNorm1d(1024),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.GELU(),
    nn.Dropout(0.5),
    nn.Linear(512, 3)
)



⸻

🔍 Key Features

✅ MC Dropout
	•	50 stochastic forward passes per image
	•	Returns mean class probability, entropy (uncertainty), and variance (optional for future)

✅ Uncertainty Metrics
	•	Predictive Entropy to quantify confidence
	•	Plans to include Variance and Uncertainty Calibration

✅ Explainability
	•	Grad-CAM overlays highlight regions that contribute most to predictions
	•	Supports visual inspection for each image

✅ Visualization Tools
	•	Confusion matrix
	•	Entropy & confidence histograms
	•	Per-class boxplots
	•	t-SNE visualizations of feature space
	•	Scatterplots of confidence vs. entropy

⸻

📊 Results

MC Dropout Evaluation:

              precision    recall  f1-score   support

      Benign       0.55      0.66      0.60        41
   Malignant       0.86      0.75      0.80        89
      Normal       0.81      0.94      0.87        18

    accuracy                           0.75       148
   macro avg       0.74      0.79      0.76       148
weighted avg       0.77      0.75      0.75       148

✅ Confusion matrix, classification report, and all uncertainty metrics are automatically generated in the analysis notebook.

⸻

🧰 Environment

See environment.yml for full dependencies.

Core Libraries:
	•	torch, torchvision, scikit-learn, seaborn, matplotlib
	•	captum for interpretability
	•	scipy.stats.entropy for uncertainty

⸻

📁 Structure

.
├── predict_with_uncertainty_and_gradcam.ipynb     # Inference + Grad-CAM script
├── train_densenet121_mc_dropout_val_confmatrix.py # Model training and evaluation
├── environment.yml                                # Conda environment file
├── data/                                          # Ultrasound images (Kaggle dataset)
├── outputs/                                       # Predictions, confusion matrix, plots
└── README.md


⸻

🧪 How to Run

1. Setup environment

conda env create -f environment.yml
conda activate bayesian-pytorch-env

2. Train the model

python train_densenet121_mc_dropout_val_confmatrix.py

3. Run single image prediction

from predict_with_uncertainty_and_gradcam import predict_with_uncertainty_and_gradcam
predict_with_uncertainty_and_gradcam("path/to/image.jpg")


⸻

🔭 Planned Improvements
	•	Replace MC Dropout with true Bayesian Neural Network (e.g. via Pyro or BNN layers)
	•	Add uncertainty calibration metrics (ECE, reliability diagrams)
	•	Deploy a web demo using Streamlit or Flask
	•	Compare entropy vs. variance-based uncertainty

⸻

📸 Sample Output
	•	Grad-CAM overlays
	•	Entropy and confidence boxplots