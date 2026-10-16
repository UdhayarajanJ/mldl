# Machine Learning & Deep Learning Lab (`mldl`)

Python implementations of core Machine Learning, Deep Learning, NLP, and Data Science algorithms built for MCA lab experiments.

## 📂 Repository Structure

This repository contains standalone practical programs covering data preprocessing, exploratory data analysis, classical machine learning models, neural networks, and natural language processing.

| **Lab / Topic** | **Key Technologies / Libraries** | **Description** | 
| **NumPy Basics** | `NumPy` | Array creation, attributes, math functions, and slicing. | 
| **Data Import/Export** | `Pandas` | Reading and writing CSV, JSON, and Excel data files. | 
| **Correlation & Heatmaps** | `Pandas`, `Seaborn` | Calculating correlation matrices and visualizing heatmaps. | 
| **Data Preprocessing** | `Scikit-Learn` | Handling missing values, feature standardization, and train-test splitting. | 
| **Data Visualization** | `Matplotlib`, `Seaborn` | Line plots, bar charts, histograms, scatter plots, and box plots. | 
| **Classifier Evaluation** | `Scikit-Learn` | Train-Test split, Accuracy, Precision, Recall, F1-Score, and Confusion Matrix. | 
| **Linear Regression** | `Scikit-Learn`, `Matplotlib` | Fitting linear models, calculating MSE/R2, and plotting regression lines. | 
| **Logistic Regression** | `Scikit-Learn`, `Matplotlib` | Binary classification, probability modeling, and Sigmoid curves. | 
| **Random Forest** | `Scikit-Learn` | Ensemble learning, bagging trees, and feature importance analysis. | 
| **Naïve Bayes** | `Scikit-Learn`, `Pandas` | Probabilistic classification from CSV files and evaluation heatmaps. | 
| **Artificial Neural Networks** | `Scikit-Learn` | Multilayer Perceptron (MLP) classification and training loss curves. | 
| **Feature Selection** | `Scikit-Learn` | SelectKBest, RFE, and SelectFromModel comparative accuracy tests. | 
| **Convolutional Neural Networks** | `TensorFlow`, `Keras` | Image classification using Conv2D, MaxPooling, and Dense layers. | 
| **Beautiful Soup Text Processing** | `BeautifulSoup4` | HTML parsing, tag searching, and text extraction. | 
| **NLTK NLP Processing** | `NLTK` | Tokenization, stop-word removal, stemming, and lemmatization. | 

## 🛠️ Prerequisites & Installation

Ensure you have Python 3.14+ installed. You can install all required libraries using pip:

```
pip install numpy pandas scikit-learn tensorflow keras nltk seaborn matplotlib beautifulsoup4 openpyxl

```

For NLTK text processing modules, ensure tokenizers and corporas are downloaded:

```
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

```

## 🚀 Running the Programs

Each lab file is structured to be run independently. For example, to run the python script:

```
python app.py

```

## 👨‍💻 Author

Built as part of the Master of Computer Applications (MCA) lab curriculum.