import streamlit as st

st.title("Hello World 🌍")
st.write("My first Streamlit deployment!")


 📘 Sleep Health Report - Regression (Supervised Learning)

# Step 1: Import libraries
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import numpy as np # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore

# Step 2: Load dataset
file_path = "Data Collection for ML mini project (Responses) - Form Responses 1.csv"
df = pd.read_csv(file_path)

# Step 3: Select required features
cols = {
    "social_media": "  Daily Social Media Minutes  \n(Provide values in integer between 0-600)",
    "gaming_hours": "  Gaming hours per week  \n(Provide Values in integer between 0-50)",
    "intro_extro": "  Introversion extraversion  "
}
data = df[[cols["social_media"], cols["gaming_hours"], cols["intro_extro"]]].copy()

# Step 4: Clean column names
data.columns = ["social_media_minutes", "gaming_hours_per_week", "introversion_extraversion"]

# Step 5: Convert to numeric (handle messy entries like '120 cm')
for col in data.columns:
    data[col] = pd.to_numeric(data[col].astype(str).str.extract("(\d+)")[0], errors="coerce")

# Step 6: Handle missing values (fill with median)
data = data.fillna(data.median())

# Step 7: Create synthetic target (sleep_hours)
# Assume: more social media/gaming → less sleep, more introversion → more sleep
np.random.seed(42)
data["sleep_hours"] = (
    10
    - 0.005 * data["social_media_minutes"]
    - 0.05 * data["gaming_hours_per_week"]
    + 0.3 * data["introversion_extraversion"]
    + np.random.normal(0, 0.5, len(data))
).clip(4, 10)  # limit between 4-10 hrs

# Step 8: Explore dataset
print(data.head())
sns.pairplot(data, diag_kind="kde")
plt.show()

# Step 9: Train-test split
X = data[["social_media_minutes", "gaming_hours_per_week", "introversion_extraversion"]]
y = data["sleep_hours"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 11: Predictions
y_pred = model.predict(X_test)

# Step 12: Evaluation
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 13: Coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print(coef_df)

# Step 14: Plot Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Sleep Hours")
plt.ylabel("Predicted Sleep Hours")
plt.title("Actual vs Predicted Sleep Hours")
plt.plot([4, 10], [4, 10], 'r--')
plt.show()
