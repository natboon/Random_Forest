import tkinter as tk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
import random

bg_color = '#2A3457'
fg_color = '#F2EAED'
highlight_color = '#FFD95A'

window = tk.Tk()
window.title("Weather Prediction")
window.geometry("1500x800")
window.config(bg=bg_color)
window.resizable(0, 0)

# Mockup data generation for demonstration purposes
np.random.seed(42)
data_size = 100
features = pd.DataFrame({
    'AAV': np.random.randint(0, 3, data_size),
    'BBL': np.random.randint(0, 142.50, data_size),
    'CENTER': np.random.randint(0, 43, data_size),
    'CPALL': np.random.randint(0, 54.50, data_size),
    'KBANK': np.random.randint(0, 122, data_size),
    'Target': 4 * np.random.rand(data_size) + 20
})

# Random Forest model training
X = features[['AAV', 'BBL', 'CENTER', 'CPALL', 'KBANK']]
y = features['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Frame for input and prediction
input_frame = ttk.Frame(window, padding="10")
input_frame.pack(side="left", fill="both", expand=True)

# Label and Entry for Features
label_features = ttk.Label(input_frame, text="Enter Share:", font=("Rockwell", 15, "bold"))
label_features.grid(row=0, column=0, columnspan=2, pady=10)

feature_entries = []
for i, feature in enumerate(X.columns):
    label = ttk.Label(input_frame, text=f"{feature}:", font=("Rockwell", 12))
    label.grid(row=i+1, column=0, sticky="e", pady=5)

    entry = ttk.Entry(input_frame, font=("Rockwell", 12))
    entry.grid(row=i+1, column=1, pady=5)
    feature_entries.append(entry)

# Button to Predict
button_predict = ttk.Button(input_frame, text="Predict", command=lambda: predict())
button_predict.grid(row=len(X.columns)+1, column=0, columnspan=2, pady=10)

# Label for Prediction Result
label_result = ttk.Label(input_frame, text="", font=("Rockwell", 15))
label_result.grid(row=len(X.columns)+2, column=0, columnspan=2, pady=10)

# Frame for displaying training data
display_frame = ttk.Frame(window, padding="10")
display_frame.pack(side="right", fill="both", expand=True)

# Treeview for Displaying Data
treeview_columns = ['AAV', 'BBL', 'CENTER', 'CPALL', 'KBANK', 'Target']
treeview = ttk.Treeview(display_frame, columns=treeview_columns, show='headings')

for col in treeview_columns:
    treeview.heading(col, text=col)

for index, row in features.iterrows():
    treeview.insert("", index, values=tuple(row))

treeview.pack(expand=tk.YES, fill=tk.BOTH)

# Function to Predict and Display Result
def predict():
    try:
        input_values = [float(entry.get()) for entry in feature_entries]
        prediction_result = model.predict([input_values])[0]
        random_feature = random.choice(X.columns)
        label_result.config(text=f"{random_feature} Prediction: {prediction_result:.2f} ")
    except ValueError:
        label_result.config(text="Invalid input. Please enter numeric values for all features.")

window.mainloop()
