import streamlit as st

st.title("😴 Sleep Schedule Prediction App")
st.write("Machine Learning Mini Project ")
st.write("""
This app predicts your **average sleep hours per night** 
based on your social media usage, gaming time, and personality type.
""")


# sleep_prediction_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Step 1: Upload or Load Dataset
st.subheader("📂 Upload Dataset or Use Default")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Data Collection for ML mini project (Responses) - Form Responses 1.csv")

# --- 1️⃣ Raw Dataset ---
st.subheader("🧾 Raw Dataset (First 5 Rows)")
st.dataframe(df.head())

# Step 2: Clean Data
cols = {
    "social_media": "  Daily Social Media Minutes  \n(Provide values in integer between 0-600)",
    "gaming_hours": "  Gaming hours per week  \n(Provide Values in integer between 0-50)",
    "intro_extro": "  Introversion extraversion  "
}

data = df[[cols["social_media"], cols["gaming_hours"], cols["intro_extro"]]].copy()
data.columns = ["daily_social_media_minutes", "gaming_hours_per_week", "introversion_extraversion"]

# Convert to numeric
for col in data.columns:
    data[col] = pd.to_numeric(data[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

# Fill missing values with median
data.fillna(data.median(), inplace=True)

# Create synthetic sleep hours target
np.random.seed(42)
data["sleep_hours"] = (
    10
    - 0.005 * data["daily_social_media_minutes"]
    - 0.05 * data["gaming_hours_per_week"]
    + 0.3 * data["introversion_extraversion"]
    + np.random.normal(0, 0.5, len(data))
).clip(4, 10)

# --- 2️⃣ Cleaned Dataset ---
st.subheader("🧹 Cleaned Dataset (First 5 Rows)")
st.dataframe(data.head())

# Step 3: Train Model
X = data[["daily_social_media_minutes", "gaming_hours_per_week", "introversion_extraversion"]]
y = data["sleep_hours"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- 3️⃣ Model Evaluation ---
st.subheader("📊 Model Evaluation")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Step 4: Descriptive Analytics
st.subheader("📈 Descriptive Analytics")
st.write("### Summary Statistics")
st.dataframe(data.describe())

st.write("### Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Step 5: Prediction Interface
st.subheader("🎯 Predict Sleep Hours")
social_media = st.number_input("Daily Social Media Minutes", 0, 600, 120)
gaming_hours = st.number_input("Gaming Hours per Week", 0, 50, 5)
intro_extro = st.number_input("Introversion-Extraversion Score", 0, 10, 5)

if st.button("Predict Sleep Schedule"):
    pred = model.predict([[social_media, gaming_hours, intro_extro]])[0]
    st.success(f"Predicted Average Sleep Hours: {pred:.2f} hours per day 😴")




import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------
# 💤 Title & Introduction
# -----------------------------------------------


# -----------------------------------------------
# 🧹 Data Preprocessing
# -----------------------------------------------
col_map = {
    "social_media": "  Daily Social Media Minutes  \n(Provide values in integer between 0-600)",
    "gaming": "  Gaming hours per week  \n(Provide Values in integer between 0-50)",
    "personality": "  Introversion extraversion  "
}

data = df[[col_map["social_media"], col_map["gaming"], col_map["personality"]]].copy()
data.columns = ["social_media_minutes", "gaming_hours_per_week", "introversion_extraversion"]

# Clean numeric data
for col in data.columns:
    data[col] = pd.to_numeric(
        data[col].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    )

data.fillna(data.median(), inplace=True)

# Create synthetic sleep hours target
np.random.seed(42)
data["sleep_hours"] = (
    10
    - 0.004 * data["social_media_minutes"]
    - 0.05 * data["gaming_hours_per_week"]
    + 0.3 * data["introversion_extraversion"]
    + np.random.normal(0, 0.4, len(data))
).clip(4, 10)

st.write("### 🧾 Processed Dataset Sample")
st.dataframe(data.head())

# -----------------------------------------------
# 📊 Train Model
# -----------------------------------------------
X = data[["social_media_minutes", "gaming_hours_per_week", "introversion_extraversion"]]
y = data["sleep_hours"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------------------------
# 📈 Model Evaluation
# -----------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📉 Model Performance")
st.write(f"**Mean Absolute Error:** {mae:.2f}")
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# -----------------------------------------------
# 🧠 User Prediction
# -----------------------------------------------
st.subheader("🕒 Predict Your Sleep Hours")

social_media = st.number_input("📱 Daily Social Media Minutes", 0, 600, 120)
gaming = st.number_input("🎮 Gaming Hours per Week", 0, 50, 10)
personality = st.slider("🧍 Introversion-Extraversion (1 = Introvert, 10 = Extrovert)", 1, 10, 5)

if st.button("Predict Sleep Duration"):
    user_input = np.array([[social_media, gaming, personality]])
    prediction = model.predict(user_input)[0]
    st.success(f"😴 You are likely to sleep around **{prediction:.2f} hours per night.**")

# -----------------------------------------------
# 📊 Visualization
# -----------------------------------------------
st.subheader("📈 Lifestyle vs Sleep Hours")

fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(
    x=data["social_media_minutes"],
    y=data["sleep_hours"],
    hue=data["introversion_extraversion"],
    palette="coolwarm",
    ax=ax
)
ax.set_xlabel("Daily Social Media Minutes")
ax.set_ylabel("Sleep Hours per Night")
ax.set_title("Sleep Hours vs Social Media Usage")
st.pyplot(fig)

# -----------------------------------------------
# ✅ End of App
# -----------------------------------------------
st.caption("Developed by Debopriya Bhattacharjee 💻 | Streamlit + ML Project")
