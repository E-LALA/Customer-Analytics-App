# Customer Analytics Streamlit App

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-🎈-red.svg)](https://streamlit.io/)
![License](https://img.shields.io/badge/License-MIT-green.svg) A web application built using Streamlit for customer analytics, enabling users to perform Churn Prediction (Naive Bayes) and Customer Segmentation (K-Means) by uploading their own datasets.

## ✨ Features

-   📊 **Customer Churn Prediction:** Predict customer churn probability using a trained Naive Bayes model.
    -   Requires an **Excel file (.xlsx)** with specific columns.
    -   Displays prediction for user input, probability plot, model accuracy, and confusion matrix.
-   🧩 **Customer Segmentation (K-Means):** Cluster customers into groups based on selected features.
    -   Requires a **CSV file (.csv)** with numeric data.
    -   Allows interactive selection of features and number of clusters (K).
    -   Visualizes clusters in 2D or 3D scatter plots using Plotly.
    -   Shows cluster distribution in a bar chart.
-   ☁️ **Data Upload:** Easy file upload interface for user datasets.
-   🖱️ **Interactive Interface:** Simple sliders, radio buttons, and dropdowns for user interaction.

## Prerequisites

-   Python 3.7+
-   `pip` package manager

## Installation

1.  Clone this repository:

    ```bash
    git clone [https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```
    (Replace `YOUR_GITHUB_USERNAME` and `YOUR_REPOSITORY_NAME` with your actual GitHub details).
    *Alternatively, if you only have the script file (`app.py`), just navigate to its directory.*

2.  Install the required Python packages:

    ```bash
    pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib openpyxl
    ```
## How to Run

1.  Make sure you are in the project directory in your terminal.
2.  Run the Streamlit application:

    ```bash
    streamlit run cutomerAnalysis.py
    ```
3.  The app will open in your default web browser at `http://localhost:8501`.

## 🚀 How to Use

1.  Open the application in your web browser.
2.  Use the sidebar on the left to select either "Customer Churn Prediction" or "Customer Segmentation (K-Means)".
3.  Follow the instructions on the selected page to upload the required data file (.xlsx for Churn, .csv for Segmentation).
4.  Interact with the input widgets in the sidebar (for Churn prediction) or select features/parameters (for Segmentation) to see the analysis results and visualizations.

## 📂 Data Requirements

To use the app with your own data, ensure your files meet the following criteria:

* **For Customer Churn Prediction (.xlsx):**
    -   Must be an Excel file (`.xlsx`).
    -   Must contain columns: `Age` (numeric), `Tenure` (numeric, e.g., months), `Sex` (text, e.g., 'Male', 'Female'), `Churn` (text, e.g., 'Yes', 'No').
    -   The app handles basic mapping and missing values removal.
* **For Customer Segmentation (.csv):**
    -   Must be a CSV file (`.csv`).
    -   Must contain at least two numeric columns suitable for clustering. Common examples: `Age`, `Annual Income (k$)`, `Spending Score (1-100)`.
    -   The app handles missing values removal and scales the selected numeric features.
