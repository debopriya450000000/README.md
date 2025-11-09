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


import pandas as pd
import numpy as np

# --- Step 1: Load dataset ---
file_path = "Data Collection for ML mini project (Responses) - Form Responses 1.csv"
df = pd.read_csv(file_path)

# --- Step 2: Select relevant columns ---
selected_cols = [
    "  Daily Social Media Minutes  \n(Provide values in integer between 0-600)",
    "  Gaming hours per week  \n(Provide Values in integer between 0-50)",
    "  Introversion extraversion  "
]

sleep_df = df[selected_cols].copy()

# --- Step 3: Clean column names ---
sleep_df.columns = [
    "daily_social_media_minutes",
    "gaming_hours_per_week",
    "introversion_extraversion"
]

# --- Step 4: Convert text values to numeric ---
for col in sleep_df.columns:
    sleep_df[col] = pd.to_numeric(
        sleep_df[col].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    )

# --- Step 5: Handle missing values (fill with median) ---
sleep_df.fillna(sleep_df.median(), inplace=True)

# --- Step 6: Create a synthetic target variable (sleep_hours) ---
# This simulates the relationship for demo purposes
np.random.seed(42)
sleep_df["sleep_hours"] = (
    10
    - 0.005 * sleep_df["daily_social_media_minutes"]
    - 0.05 * sleep_df["gaming_hours_per_week"]
    + 0.3 * sleep_df["introversion_extraversion"]
    + np.random.normal(0, 0.5, len(sleep_df))
).clip(4, 10)  # Sleep between 4 and 10 hours

# --- Step 7: Check data ---
print(sleep_df.head())
print("\nDataset shape:", sleep_df.shape)


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("💤 Sleep Health Prediction App")

st.write("Upload your dataset or use the default one to predict average sleep hours per day.")

# --- Upload CSV (with unique key to avoid duplicate ID error) ---
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="sleep_data_upload")

# --- Load dataset ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    st.info("Using default dataset.")
    df = pd.read_csv("Data Collection for ML mini project (Responses) - Form Responses 1.csv")

# --- Select relevant columns ---
selected_cols = [
    "  Daily Social Media Minutes  \n(Provide values in integer between 0-600)",
    "  Gaming hours per week  \n(Provide Values in integer between 0-50)",
    "  Introversion extraversion  "
]

sleep_df = df[selected_cols].copy()
sleep_df.columns = [
    "daily_social_media_minutes",
    "gaming_hours_per_week",
    "introversion_extraversion"
]

# --- Data cleaning ---
for col in sleep_df.columns:
    sleep_df[col] = pd.to_numeric(
        sleep_df[col].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    )

sleep_df.fillna(sleep_df.median(), inplace=True)

# --- Create synthetic target ---
np.random.seed(42)
sleep_df["sleep_hours"] = (
    10
    - 0.005 * sleep_df["daily_social_media_minutes"]
    - 0.05 * sleep_df["gaming_hours_per_week"]
    + 0.3 * sleep_df["introversion_extraversion"]
    + np.random.normal(0, 0.5, len(sleep_df))
).clip(4, 10)

# --- Model training ---
X = sleep_df[["daily_social_media_minutes", "gaming_hours_per_week", "introversion_extraversion"]]
y = sleep_df["sleep_hours"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# --- User Input for Prediction ---
st.subheader("Try Your Own Prediction 👇")

social_media = st.number_input("Daily Social Media Minutes", min_value=0, max_value=600, value=120)
gaming = st.number_input("Gaming Hours per Week", min_value=0, max_value=50, value=10)
intro_extro = st.slider("Introversion-Extraversion (1 = Introvert, 10 = Extrovert)", 1, 10, 5)

# --- Predict ---
user_input = np.array([[social_media, gaming, intro_extro]])
predicted_sleep = model.predict(user_input)[0]

st.success(f"🛌 Predicted Average Sleep Hours: **{predicted_sleep:.2f} hours/night**")

# --- Show sample data ---
with st.expander("📊 View Processed Dataset"):
    st.dataframe(sleep_df.head())


