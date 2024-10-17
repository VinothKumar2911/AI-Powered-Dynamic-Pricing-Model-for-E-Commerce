from flask import Flask, request, jsonify
import numpy as np
import pickle
import mysql.connector
from flask_cors import CORS
app = Flask(__name__)
CORS(app) 

with open('pricing_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def connect_db():
    conn = mysql.connector.connect(
        host='localhost',  
        user='root',  
        password='root',  
        database='Demo'  
    )
    return conn


@app.route('/predict_price', methods=['POST'])
def predict_price():
    data = request.get_json()
    views = data['views']
    competitor_price = data['competitor_price']
    stock_level = data['stock_level']
    time_of_day = data['time_of_day']

    
    predicted_price = model.predict([[views, competitor_price, stock_level, time_of_day]])
    
    
    return jsonify({'predicted_price': round(predicted_price[0], 2)})


@app.route('/products/predicted_prices', methods=['GET'])
def get_products_with_predictions():
    conn = connect_db()
    cursor = conn.cursor()
   
    cursor.execute("SELECT * FROM products")
    products = cursor.fetchall()

    product_list = []
    for p in products:
        views, competitor_price, stock_level, time_of_day = p[1], p[2], p[3], p[4]
        
        
        predicted_price = model.predict([[views, competitor_price, stock_level, time_of_day]])
        
        
        product_list.append({
            'id': p[0],
            'views': views,
            'competitor_price': competitor_price,
            'stock_level': stock_level,
            'time_of_day': time_of_day,
            'predicted_price': round(predicted_price[0], 2)  
        })

    conn.close()
    return jsonify(product_list)

if __name__ == '__main__':
    app.run(debug=True)
