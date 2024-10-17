import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


data = {
    'views': [120, 80, 200, 150, 100],
    'competitor_price': [100, 110, 90, 95, 105],
    'stock_level': [10, 20, 15, 8, 12],
    'time_of_day': [14, 18, 20, 16, 12],  # Assume 24-hour format
    'price': [105, 100, 110, 98, 102]
}

df = pd.DataFrame(data)


X = df[['views', 'competitor_price', 'stock_level', 'time_of_day']]
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


with open('pricing_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


print("Model trained and saved successfully!")
