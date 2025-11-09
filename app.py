import streamlit as st

st.title("Hello World 🌍")
st.write("My first Streamlit deployment!")


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
st.title("🧠 Sleep Health Prediction App")

st.subheader("Upload your dataset or use default")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    file_path = "Data Collection for ML mini project (Responses) - Form Responses 1.csv"
    df = pd.read_csv(file_path)

st.write("### Dataset Preview:")
st.dataframe(df.head())
cols = {
    "social_media": "  Daily Social Media Minutes  \n(Provide values in integer between 0-600)",
    "gaming_hours": "  Gaming hours per week  \n(Provide Values in integer between 0-50)",
    "intro_extro": "  Introversion extraversion  "
}

data = df[[cols["social_media"], cols["gaming_hours"], cols["intro_extro"]]].copy()
data.columns = ["social_media_minutes", "gaming_hours_per_week", "introversion_extraversion"]

# Convert messy entries (like "120 mins") into numbers
for col in data.columns:
    data[col] = pd.to_numeric(data[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

# Fill missing values
data = data.fillna(data.median())

st.write("### Cleaned Data:")
st.dataframe(data.head())
# Split data (80% training, 20% testing)
X = data[["social_media_minutes", "gaming_hours_per_week"]]
y = data["introversion_extraversion"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Mean Absolute Error:** {mae:.2f}")
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")
st.subheader("🎯 Predict Introversion/Extraversion")

social_media = st.number_input("Daily Social Media Minutes", 0, 600, 120)
gaming_hours = st.number_input("Gaming Hours per Week", 0, 50, 5)

if st.button("Predict Personality Score"):
    pred = model.predict([[social_media, gaming_hours]])[0]
    st.success(f"Predicted Personality Score: {pred:.2f}")
st.subheader("📈 Relationship between Features")
fig, ax = plt.subplots()
sns.scatterplot(x=data["social_media_minutes"], y=data["introversion_extraversion"], ax=ax)
ax.set_xlabel("Social Media Minutes")
ax.set_ylabel("Personality Score")
st.pyplot(fig)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("🧠 Sleep Health Prediction App")

st.subheader("Upload your dataset or use default")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    file_path = "Data Collection for ML mini project (Responses) - Form Responses 1.csv"
    df = pd.read_csv(file_path)

st.write("### Dataset Preview:")
st.dataframe(df.head())

# Clean and preprocess
cols = {
    "social_media": "  Daily Social Media Minutes  \n(Provide values in integer between 0-600)",
    "gaming_hours": "  Gaming hours per week  \n(Provide Values in integer between 0-50)",
    "intro_extro": "  Introversion extraversion  "
}

data = df[[cols["social_media"], cols["gaming_hours"], cols["intro_extro"]]].copy()
data.columns = ["social_media_minutes", "gaming_hours_per_week", "introversion_extraversion"]

for col in data.columns:
    data[col] = pd.to_numeric(data[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

data = data.fillna(data.median())

st.write("### Cleaned Data:")
st.dataframe(data.head())

# Train model
X = data[["social_media_minutes", "gaming_hours_per_week"]]
y = data["introversion_extraversion"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Mean Absolute Error:** {mae:.2f}")
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Prediction UI
st.subheader("🎯 Predict Introversion/Extraversion")
social_media = st.number_input("Daily Social Media Minutes", 0, 600, 120)
gaming_hours = st.number_input("Gaming Hours per Week", 0, 50, 5)

if st.button("Predict Personality Score"):
    pred = model.predict([[social_media, gaming_hours]])[0]
    st.success(f"Predicted Personality Score: {pred:.2f}")

# Visualization
st.subheader("📈 Relationship between Features")
fig, ax = plt.subplots()
sns.scatterplot(x=data["social_media_minutes"], y=data["introversion_extraversion"], ax=ax)
ax.set_xlabel("Social Media Minutes")
ax.set_ylabel("Personality Score")
st.pyplot(fig)
