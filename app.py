import streamlit as st

st.title("Hello World 🌍")
st.write("My first Streamlit deployment!")
import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  # or your actual model

# --- Step 1: Load dataset and train model ---
df = pd.read_csv("Data Collection for ML mini project (Responses) - Form Responses 1.csv")

# Your preprocessing steps (same as in your notebook)
# Example:
X = df[['Age', 'Sleep Duration', 'Stress Level']]  # example columns
y = df['Quality of Sleep']

model = RandomForestClassifier()
model.fit(X, y)

# --- Step 2: Streamlit UI ---
st.title("🛏️ Sleep Schedule Prediction App")
st.write("Predict your sleep quality based on your daily routine!")

age = st.number_input("Enter your age:", min_value=0, max_value=100)
sleep_duration = st.number_input("Enter your sleep duration (hours):", min_value=0.0, max_value=12.0)
stress_level = st.slider("Stress Level (1-10):", 1, 10)

# --- Step 3: Make Prediction ---
if st.button("Predict"):
    input_data = [[age, sleep_duration, stress_level]]
    prediction = model.predict(input_data)
    st.success(f"Predicted Sleep Quality: {prediction[0]}")


import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  # or your actual model

# --- Step 1: Load dataset and train model ---
df = pd.read_csv("Data Collection for ML mini project (Responses) - Form Responses 1.csv")

# Your preprocessing steps (same as in your notebook)
# Example:
X = df[['Age', 'Sleep Duration', 'Stress Level']]  # example columns
y = df['Quality of Sleep']

model = RandomForestClassifier()
model.fit(X, y)

# --- Step 2: Streamlit UI ---
st.title("🛏️ Sleep Schedule Prediction App")
st.write("Predict your sleep quality based on your daily routine!")

age = st.number_input("Enter your age:", min_value=0, max_value=100)
sleep_duration = st.number_input("Enter your sleep duration (hours):", min_value=0.0, max_value=12.0)
stress_level = st.slider("Stress Level (1-10):", 1, 10)

# --- Step 3: Make Prediction ---
if st.button("Predict"):
    input_data = [[age, sleep_duration, stress_level]]
    prediction = model.predict(input_data)
    st.success(f"Predicted Sleep Quality: {prediction[0]}")


import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  # or your actual model

# --- Step 1: Load dataset and train model ---
df = pd.read_csv("Data Collection for ML mini project (Responses) - Form Responses 1.csv")

# Your preprocessing steps (same as in your notebook)
# Example:
X = df[['Age', 'Sleep Duration', 'Stress Level']]  # example columns
y = df['Quality of Sleep']

model = RandomForestClassifier()
model.fit(X, y)

# --- Step 2: Streamlit UI ---
st.title("🛏️ Sleep Schedule Prediction App")
st.write("Predict your sleep quality based on your daily routine!")

age = st.number_input("Enter your age:", min_value=0, max_value=100)
sleep_duration = st.number_input("Enter your sleep duration (hours):", min_value=0.0, max_value=12.0)
stress_level = st.slider("Stress Level (1-10):", 1, 10)

# --- Step 3: Make Prediction ---
if st.button("Predict"):
    input_data = [[age, sleep_duration, stress_level]]
    prediction = model.predict(input_data)
    st.success(f"Predicted Sleep Quality: {prediction[0]}")


import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  # or your actual model

# --- Step 1: Load dataset and train model ---
df = pd.read_csv("Data Collection for ML mini project (Responses) - Form Responses 1.csv")

# Your preprocessing steps (same as in your notebook)
# Example:
X = df[['Age', 'Sleep Duration', 'Stress Level']]  # example columns
y = df['Quality of Sleep']

model = RandomForestClassifier()
model.fit(X, y)

# --- Step 2: Streamlit UI ---
st.title("🛏️ Sleep Schedule Prediction App")
st.write("Predict your sleep quality based on your daily routine!")

age = st.number_input("Enter your age:", min_value=0, max_value=100)
sleep_duration = st.number_input("Enter your sleep duration (hours):", min_value=0.0, max_value=12.0)
stress_level = st.slider("Stress Level (1-10):", 1, 10)

# --- Step 3: Make Prediction ---
if st.button("Predict"):
    input_data = [[age, sleep_duration, stress_level]]
    prediction = model.predict(input_data)
    st.success(f"Predicted Sleep Quality: {prediction[0]}")



import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  # or your actual model

# --- Step 1: Load dataset and train model ---
df = pd.read_csv("Data Collection for ML mini project (Responses) - Form Responses 1.csv")

# Your preprocessing steps (same as in your notebook)
# Example:
X = df[['Age', 'Sleep Duration', 'Stress Level']]  # example columns
y = df['Quality of Sleep']

model = RandomForestClassifier()
model.fit(X, y)

# --- Step 2: Streamlit UI ---
st.title("🛏️ Sleep Schedule Prediction App")
st.write("Predict your sleep quality based on your daily routine!")

age = st.number_input("Enter your age:", min_value=0, max_value=100)
sleep_duration = st.number_input("Enter your sleep duration (hours):", min_value=0.0, max_value=12.0)
stress_level = st.slider("Stress Level (1-10):", 1, 10)

# --- Step 3: Make Prediction ---
if st.button("Predict"):
    input_data = [[age, sleep_duration, stress_level]]
    prediction = model.predict(input_data)
    st.success(f"Predicted Sleep Quality: {prediction[0]}")


import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  # or your actual model

# --- Step 1: Load dataset and train model ---
df = pd.read_csv("Data Collection for ML mini project (Responses) - Form Responses 1.csv")

# Your preprocessing steps (same as in your notebook)
# Example:
X = df[['Age', 'Sleep Duration', 'Stress Level']]  # example columns
y = df['Quality of Sleep']

model = RandomForestClassifier()
model.fit(X, y)

# --- Step 2: Streamlit UI ---
st.title("🛏️ Sleep Schedule Prediction App")
st.write("Predict your sleep quality based on your daily routine!")

age = st.number_input("Enter your age:", min_value=0, max_value=100)
sleep_duration = st.number_input("Enter your sleep duration (hours):", min_value=0.0, max_value=12.0)
stress_level = st.slider("Stress Level (1-10):", 1, 10)

# --- Step 3: Make Prediction ---
if st.button("Predict"):
    input_data = [[age, sleep_duration, stress_level]]
    prediction = model.predict(input_data)
    st.success(f"Predicted Sleep Quality: {prediction[0]}")


import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  # or your actual model

# --- Step 1: Load dataset and train model ---
df = pd.read_csv("Data Collection for ML mini project (Responses) - Form Responses 1.csv")

# Your preprocessing steps (same as in your notebook)
# Example:
X = df[['Age', 'Sleep Duration', 'Stress Level']]  # example columns
y = df['Quality of Sleep']

model = RandomForestClassifier()
model.fit(X, y)

# --- Step 2: Streamlit UI ---
st.title("🛏️ Sleep Schedule Prediction App")
st.write("Predict your sleep quality based on your daily routine!")

age = st.number_input("Enter your age:", min_value=0, max_value=100)
sleep_duration = st.number_input("Enter your sleep duration (hours):", min_value=0.0, max_value=12.0)
stress_level = st.slider("Stress Level (1-10):", 1, 10)

# --- Step 3: Make Prediction ---
if st.button("Predict"):
    input_data = [[age, sleep_duration, stress_level]]
    prediction = model.predict(input_data)
    st.success(f"Predicted Sleep Quality: {prediction[0]}")


import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  # or your actual model

# --- Step 1: Load dataset and train model ---
df = pd.read_csv("Data Collection for ML mini project (Responses) - Form Responses 1.csv")

# Your preprocessing steps (same as in your notebook)
# Example:
X = df[['Age', 'Sleep Duration', 'Stress Level']]  # example columns
y = df['Quality of Sleep']

model = RandomForestClassifier()
model.fit(X, y)

# --- Step 2: Streamlit UI ---
st.title("🛏️ Sleep Schedule Prediction App")
st.write("Predict your sleep quality based on your daily routine!")

age = st.number_input("Enter your age:", min_value=0, max_value=100)
sleep_duration = st.number_input("Enter your sleep duration (hours):", min_value=0.0, max_value=12.0)
stress_level = st.slider("Stress Level (1-10):", 1, 10)

# --- Step 3: Make Prediction ---
if st.button("Predict"):
    input_data = [[age, sleep_duration, stress_level]]
    prediction = model.predict(input_data)
    st.success(f"Predicted Sleep Quality: {prediction[0]}")


import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  # or your actual model

# --- Step 1: Load dataset and train model ---
df = pd.read_csv("Data Collection for ML mini project (Responses) - Form Responses 1.csv")

# Your preprocessing steps (same as in your notebook)
# Example:
X = df[['Age', 'Sleep Duration', 'Stress Level']]  # example columns
y = df['Quality of Sleep']

model = RandomForestClassifier()
model.fit(X, y)

# --- Step 2: Streamlit UI ---
st.title("🛏️ Sleep Schedule Prediction App")
st.write("Predict your sleep quality based on your daily routine!")

age = st.number_input("Enter your age:", min_value=0, max_value=100)
sleep_duration = st.number_input("Enter your sleep duration (hours):", min_value=0.0, max_value=12.0)
stress_level = st.slider("Stress Level (1-10):", 1, 10)

# --- Step 3: Make Prediction ---
if st.button("Predict"):
    input_data = [[age, sleep_duration, stress_level]]
    prediction = model.predict(input_data)
    st.success(f"Predicted Sleep Quality: {prediction[0]}")



import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  # or your actual model

# --- Step 1: Load dataset and train model ---
df = pd.read_csv("Data Collection for ML mini project (Responses) - Form Responses 1.csv")

# Your preprocessing steps (same as in your notebook)
# Example:
X = df[['Age', 'Sleep Duration', 'Stress Level']]  # example columns
y = df['Quality of Sleep']

model = RandomForestClassifier()
model.fit(X, y)

# --- Step 2: Streamlit UI ---
st.title("🛏️ Sleep Schedule Prediction App")
st.write("Predict your sleep quality based on your daily routine!")

age = st.number_input("Enter your age:", min_value=0, max_value=100)
sleep_duration = st.number_input("Enter your sleep duration (hours):", min_value=0.0, max_value=12.0)
stress_level = st.slider("Stress Level (1-10):", 1, 10)

# --- Step 3: Make Prediction ---
if st.button("Predict"):
    input_data = [[age, sleep_duration, stress_level]]
    prediction = model.predict(input_data)
    st.success(f"Predicted Sleep Quality: {prediction[0]}")
