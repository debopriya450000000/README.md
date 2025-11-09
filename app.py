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

st.title("😴 Sleep Health Prediction App")

st.write("Predict average **sleep hours per night** based on your lifestyle habits.")

# --- Upload CSV (with unique key to avoid duplication error) ---
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="sleep_data_upload")

# --- Load dataset ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    st.info("Using default dataset.")
    df = pd.read_csv("Data Collection for ML mini project (Responses) - Form Responses 1.csv")

# --- Select and rename columns ---
col_map = {
    "social_media": "  Daily Social Media Minutes  \n(Provide values in integer between 0-600)",
    "gaming": "  Gaming hours per week  \n(Provide Values in integer between 0-50)",
    "personality": "  Introversion extraversion  "
}

data = df[[col_map["social_media"], col_map["gaming"], col_map["personality"]]].copy()
data.columns = ["social_media_minutes", "gaming_hours_per_week", "introversion_extraversion"]

# --- Clean numeric data ---
for col in data.columns:
    data[col] = pd.to_numeric(
        data[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )

data.fillna(data.median(), inplace=True)

# --- Create synthetic Sleep Hours (target variable) ---
# Using a simple assumption-based linear relationship
np.random.seed(42)
data["sleep_hours"] = (
    10
    - 0.004 * data["social_media_minutes"]
    - 0.06 * data["gaming_hours_per_week"]
    + 0.25 * data["introversion_extraversion"]
    + np.random.normal(0, 0.4, len(data))
).clip(4, 10)

st.write("### 🧹 Cleaned & Processed Data:")
st.dataframe(data.head())

# --- Train-Test Split ---
X = data[["social_media_minutes", "gaming_hours_per_week", "introversion_extraversion"]]
y = data["sleep_hours"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Evaluation Metrics ---
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Mean Absolute Error:** {mae:.2f}")
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# --- Prediction Interface ---
st.subheader("💭 Predict Your Sleep Duration")

social_media = st.number_input("Daily Social Media Minutes", 0, 600, 120)
gaming = st.number_input("Gaming Hours per Week", 0, 50, 10)
personality = st.slider("Introversion-Extraversion (1 = Introvert, 10 = Extrovert)", 1, 10, 5)

if st.button("Predict Sleep Hours"):
    user_input = np.array([[social_media, gaming, personality]])
    predicted_sleep = model.predict(user_input)[0]
    st.success(f"🛌 You are likely to sleep around **{predicted_sleep:.2f} hours per night**")

# --- Visualization ---
st.subheader("📈 Relationship between Lifestyle and Sleep Hours")

fig, ax = plt.subplots()
sns.scatterplot(x=data["social_media_minutes"], y=data["sleep_hours"], ax=ax)
ax.set_xlabel("Daily Social Media Minutes")
ax.set_ylabel("Sleep Hours per Night")
st.pyplot(fig)
