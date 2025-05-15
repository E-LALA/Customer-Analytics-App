import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import io # Import io module to read file in memory

st.set_page_config(page_title="Customer ML Dashboard", layout="wide")

# --- Helper function to load and preprocess churn data ---
@st.cache_data
def process_churn_data(uploaded_file):
    """Loads, preprocesses, and cleans the churn dataset."""
    if uploaded_file is not None:
        try:
            # Read the excel file
            df = pd.read_excel(uploaded_file)

            # Check for required columns
            required_cols = ['Age', 'Tenure', 'Sex', 'Churn']
            if not all(col in df.columns for col in required_cols):
                st.error(f"The uploaded file must contain the following columns: {', '.join(required_cols)}")
                return None

            # Data preprocessing: Map 'Sex' and 'Churn' to numeric
            # Ensure mapping keys match the data case/spelling
            df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
            df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

            # Drop rows where mapping resulted in NaN (unexpected values) or other NaNs
            df.dropna(inplace=True)

            # Ensure numeric types after mapping and dropping NaNs
            for col in ['Age', 'Tenure', 'Sex', 'Churn']:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True) # Drop rows where conversion failed


            return df
        except Exception as e:
            st.error(f"Error processing the Churn file: {e}")
            return None
    return None

# --- Helper function to load and preprocess segmentation data ---
@st.cache_data
def process_segment_data(uploaded_file):
    """Loads, preprocesses, and cleans the segmentation dataset."""
    if uploaded_file is not None:
        try:
            # Read the csv file
            df = pd.read_csv(uploaded_file)

            # Check for commonly used required columns for segmentation
            required_cols_check = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
            # Check if at least two of the commonly used columns exist
            if sum(col in df.columns for col in required_cols_check) < 2:
                 st.warning(f"The uploaded segmentation file should ideally contain columns like: {', '.join(required_cols_check)}. Please ensure sufficient numeric columns are present for clustering.")
                 # Continue, but segmentation features will be limited by available numeric columns

            # Drop rows with missing values
            df.dropna(inplace=True)

            # Ensure numeric types for all potential numeric columns
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            for col in numeric_cols:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True) # Drop rows where conversion failed

            if df.empty:
                 st.error("No valid numeric data remaining after processing the segmentation file. Please check the file content.")
                 return None


            return df
        except Exception as e:
            st.error(f"Error processing the Segmentation file: {e}")
            return None
    return None


# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Customer Churn Prediction", "Customer Segmentation (K-Means)"])

# ----------------------------------------
# Page 1: Customer Churn Prediction
# ----------------------------------------
if page == "Customer Churn Prediction":
    st.title("📉 Customer Churn Prediction")
    st.markdown("Predict whether a customer will churn using a Naive Bayes model. **Please upload an Excel file.**")

    st.sidebar.subheader("Upload Data File")
    uploaded_churn_file = st.sidebar.file_uploader("Upload Customer Data (Excel file)", type=["xlsx"])

    df_churn = process_churn_data(uploaded_churn_file)

    if df_churn is not None:
        st.subheader("Overview of Uploaded Data")
        st.dataframe(df_churn.head())

        # Check if essential columns exist after processing
        if not all(col in df_churn.columns for col in ['Age', 'Tenure', 'Sex', 'Churn']):
             st.error("Processed data is missing essential columns (Age, Tenure, Sex, Churn). Please check the uploaded file.")
        elif len(df_churn) < 2:
             st.warning("Not enough data points in the processed file to train the model. Need at least 2 rows.")
        elif len(df_churn['Churn'].unique()) < 2:
             st.warning("The 'Churn' column must contain at least two unique values (Yes/No or 1/0) to train the model.")
        else:
            st.sidebar.subheader("User Input for Prediction")
            # Use min/max from the loaded data for sliders
            min_age, max_age = int(df_churn['Age'].min()), int(df_churn['Age'].max())
            min_tenure, max_tenure = int(df_churn['Tenure'].min()), int(df_churn['Tenure'].max())

            # Add checks to prevent errors if min/max are the same (e.g., only one data point)
            age_default = int(np.mean(df_churn['Age'])) if not df_churn['Age'].empty else 30
            tenure_default = int(np.mean(df_churn['Tenure'])) if not df_churn['Tenure'].empty else 12

            age = st.sidebar.slider("Age", min_age, max_age, age_default) if min_age != max_age else st.sidebar.number_input("Age", min_value=min_age, max_value=max_age, value=min_age)
            tenure = st.sidebar.slider("Tenure (Months)", min_tenure, max_tenure, tenure_default) if min_tenure != max_tenure else st.sidebar.number_input("Tenure (Months)", min_value=min_tenure, max_value=max_tenure, value=min_tenure)


            gender = st.sidebar.radio("Gender", ["Male", "Female"])
            gender_mapped = 1 if gender == "Male" else 0 # Map based on how you mapped it in process_churn_data

            # Prepare training data
            try:
                X = df_churn[['Age', 'Tenure', 'Sex']]
                y = df_churn['Churn']

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # stratify helps with imbalanced classes

                # Train model
                model = GaussianNB()
                model.fit(X_train, y_train)

                # Predict user input
                user_input = np.array([[age, tenure, gender_mapped]])
                prediction = model.predict(user_input)[0]
                probability = model.predict_proba(user_input)[0]

                st.subheader("User Input Summary")
                st.write({'Age': age, 'Tenure': tenure, 'Gender': gender})

                st.subheader("Prediction")
                st.success("Customer will NOT churn ❎" if prediction == 0 else "Customer is likely to CHURN ⚠️")

                st.subheader("Prediction Probability")
                fig = px.bar(
                    x=['No Churn', 'Churn'],
                    y=probability,
                    color=['No Churn', 'Churn'],
                    color_discrete_sequence=['#2ECC71', '#E74C3C'],
                    labels={'x': 'Prediction Outcome', 'y': 'Probability'},
                    title="Prediction Probability"
                )
                fig.update_layout(showlegend=False) # Hide legend for clarity
                st.plotly_chart(fig)

                st.subheader("Model Accuracy")
                # Evaluate model on the test set
                acc = accuracy_score(y_test, model.predict(X_test))
                st.info(f"Model Accuracy on Test Set: **{acc * 100:.2f}%**")

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, model.predict(X_test))
                fig_cm, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Actual No Churn', 'Actual Churn'], yticklabels=['Predicted No Churn', 'Predicted Churn'])
                plt.ylabel('Predicted Label')
                plt.xlabel('True Label')
                st.pyplot(fig_cm)

            except ValueError as ve:
                 st.error(f"Error preparing data or training the model: {ve}. Please ensure the data contains valid values.")
            except Exception as e:
                 st.error(f"An unexpected error occurred: {e}")

    else:
        st.info("Please upload a Customer Data file in Excel format to start churn prediction.")


# ----------------------------------------
# Page 2: Customer Segmentation
# ----------------------------------------
elif page == "Customer Segmentation (K-Means)":
    st.title("🧩 Customer Segmentation using K-Means")
    st.markdown("This section clusters customers using the K-Means algorithm. **Please upload a CSV file.**")

    st.sidebar.subheader("Upload Data File")
    uploaded_segment_file = st.sidebar.file_uploader("Upload Customer Data (CSV file)", type=["csv"])

    df_segment = process_segment_data(uploaded_segment_file)

    if df_segment is not None:
        st.subheader("Overview of Uploaded Data")
        st.dataframe(df_segment.head())

        st.subheader("Select Features for Clustering")

        # Get available numeric features from the loaded data
        available_features = df_segment.select_dtypes(include=np.number).columns.tolist()

        if not available_features:
             st.warning("No numeric columns found in the uploaded file suitable for clustering.")
        else:
            # Suggest commonly used features if available, otherwise use any 2+ numeric columns
            suggested_default = [f for f in ['Annual Income (k$)', 'Spending Score (1-100)'] if f in available_features]
            if len(suggested_default) < 2 and len(available_features) >= 2:
                 suggested_default = available_features[:2] # Use first two numeric cols if common ones not found

            selected_features = st.multiselect(
                "Select features for clustering:",
                available_features,
                default=suggested_default
            )

            if len(selected_features) >= 2:
                try:
                    X = df_segment[selected_features]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    st.subheader("Select Number of Clusters (K)")
                    # Determine a reasonable max K based on data size and number of features
                    # Max K should be less than the number of samples and ideally related to the number of features
                    max_k_samples = len(X_scaled) // 2 # Cannot have more clusters than half the samples
                    max_k_heuristic = 10 # A common heuristic limit
                    max_k = min(max_k_samples, max_k_heuristic)

                    if max_k < 2:
                         st.warning("Not enough data points or features to create clusters. Need at least 4 data points and 2 features.")
                    else:
                        k = st.slider("Choose number of clusters (K)", 2, max_k, min(5, max_k)) # Default K=5, capped by max_k

                        # Perform K-Means clustering
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # Added n_init for KMeans robustness
                        df_segment['Cluster'] = kmeans.fit_predict(X_scaled)

                        st.subheader("Visualize Clusters")

                        if len(selected_features) == 2:
                             # 2D Scatter plot
                            fig = px.scatter(
                                df_segment,
                                x=selected_features[0],
                                y=selected_features[1],
                                color=df_segment['Cluster'].astype(str),
                                title=f"Customer Clusters based on {selected_features[0]} vs {selected_features[1]}",
                                labels={selected_features[0]: selected_features[0], selected_features[1]: selected_features[1]}, # Use column names as labels
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                            st.plotly_chart(fig)
                        elif len(selected_features) == 3:
                            # 3D Scatter plot
                             fig = px.scatter_3d(
                                df_segment,
                                x=selected_features[0],
                                y=selected_features[1],
                                z=selected_features[2],
                                color=df_segment['Cluster'].astype(str),
                                title=f"Customer Clusters based on {', '.join(selected_features)}",
                                labels={
                                    selected_features[0]: selected_features[0],
                                    selected_features[1]: selected_features[1],
                                    selected_features[2]: selected_features[2]
                                },
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                             st.plotly_chart(fig)
                        else:
                             st.warning("Please select 2 or 3 features to visualize the clusters in a scatter plot.")


                        st.subheader("Cluster Distribution")
                        cluster_counts = df_segment['Cluster'].value_counts().sort_index()
                        fig2 = px.bar(
                            x=cluster_counts.index,
                            y=cluster_counts.values,
                            labels={'x': 'Cluster', 'y': 'Count'},
                            color=cluster_counts.index.astype(str),
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            title="Distribution of Customers per Cluster"
                        )
                        fig2.update_layout(xaxis={'categoryorder':'category ascending'}) # Ensure clusters are in order 0, 1, 2...
                        st.plotly_chart(fig2)

                except ValueError as ve:
                     st.error(f"Error processing data or applying K-Means: {ve}. Please ensure selected features contain valid numeric data.")
                except Exception as e:
                     st.error(f"An unexpected error occurred during customer segmentation: {e}")

            else:
                st.warning("Please select at least 2 features to perform K-Means clustering.")

    else:
        st.info("Please upload a Customer Data file in CSV format to start customer segmentation.")