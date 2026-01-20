import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("Cancer_Data.csv")

# Title
st.title("Breast Cancer Data Dashboard")

# Sidebar
st.sidebar.header("Dashboard Options")
show_overview = st.sidebar.checkbox("Show Data Overview", value=True)
show_visualizations = st.sidebar.checkbox("Show Visualizations", value=True)
show_predictions = st.sidebar.checkbox("Show Predictions", value=False)

if show_overview:
    st.header("Data Overview")
    st.write(f"Dataset shape: {df.shape}")
    st.dataframe(df.head())
    st.write("Data Info:")
    st.text(df.describe())

if show_visualizations:
    st.header("Visualizations")
    
    # Diagnosis distribution
    st.subheader("Diagnosis Distribution")
    fig = px.histogram(df, x='diagnosis', title="Diagnosis Count")
    st.plotly_chart(fig)
    
    # Feature histograms
    st.subheader("Feature Distributions")
    features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
    for feature in features:
        fig = px.histogram(df, x=feature, color='diagnosis', title=f"{feature} Distribution")
        st.plotly_chart(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, ax=ax)
    st.pyplot(fig)
    
    # Scatter plot
    st.subheader("Scatter Plot")
    fig = px.scatter(df, x='radius_mean', y='area_mean', color='diagnosis', title="Radius Mean vs Area Mean")
    st.plotly_chart(fig)

if show_predictions:
    st.header("Predictions")
    # Prepare data
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
    # User input for prediction
    st.subheader("Predict Diagnosis")
    input_data = {}
    for col in X.columns[:5]:  # First 5 features for simplicity
        input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))
    
    if st.button("Predict"):
        prediction = model.predict(pd.DataFrame([input_data]))
        result = "Malignant" if prediction[0] == 1 else "Benign"
        st.write(f"Predicted Diagnosis: {result}")
