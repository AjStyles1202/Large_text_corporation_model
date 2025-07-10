# Large_text_corporation_model

This project applies unsupervised learning (clustering) on a real-world **SMS Spam Detection Dataset** to group similar messages and uncover latent patterns without supervision. The goal is to cluster a large corpus of short text messages into meaningful groups (e.g., spam vs ham), visualize them, and analyze the cluster characteristics.

---

## 🚀 Project Overview

- **Dataset**: [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Goal**: Group similar messages using clustering algorithms like **KMeans**, and visualize groupings using **PCA**.
- **Techniques Used**:
  - Text cleaning and preprocessing
  - Feature extraction using **TF-IDF**
  - Unsupervised clustering via **KMeans**
  - Optimal `k` selection using **Silhouette Score**
  - Visual analysis with **PCA**
  - Cluster labeling analysis against original spam/ham labels

---

## 🛠️ Technologies & Libraries

- Python 3.9+
- `scikit-learn` (TF-IDF, KMeans, silhouette score, PCA)
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `re` for text cleaning

- 
## 📁 Project Structure
├── SMSSpamCollection # Raw dataset
├── sms_cluster_results.csv # Final clustered output
├── sms_clustering_model.ipynb # Full code for training and clustering
├── README.md # You're here

## 🧠 Key Steps

1. **Data Loading**  
   Read and inspect ~5,500 text messages labeled as spam or ham.

2. **Preprocessing**  
   Clean text (lowercase, remove punctuation, remove stopwords, basic lemmatization).

3. **Feature Extraction**  
   Use `TfidfVectorizer` to transform messages into numerical vectors.

4. **Model Training**  
   - Train-test split applied.
   - Use **KMeans** clustering with optimal `k` from Silhouette Score.
   - Predict test clusters and compare to original labels.

5. **Evaluation**  
   Use `pandas.crosstab` to analyze cluster-class alignment.

6. **Visualization**  
   Reduce dimensionality using **PCA** and plot clusters in 2D.

7. **Export**  
   Final results exported to `sms_cluster_results.csv`.

---

## 📷 Visual Output

- **Silhouette Score vs Number of Clusters**
- **PCA Visualization of Clusters**
  
> All plots generated via `matplotlib` and easy to reproduce.

---

## 📌 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/sms-clustering-model.git
cd sms-clustering-model

# 2. Run the model after creating the model
python sms_clustering_model.ipynb
