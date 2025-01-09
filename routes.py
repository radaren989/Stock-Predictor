from flask import Blueprint, render_template, request
from Predictor import StockPricePredictor
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
import matplotlib

stockPricePredictor = StockPricePredictor()

matplotlib.use('Agg')  # GUI hatalarını engellemek için Agg backend'ini kullan

# Blueprint oluşturuyoruz
app_routes = Blueprint('app_routes', __name__)

# Modeli yükle veya eğit
model_path = 'stock_predictor_model.h5'

# Routes
@app_routes.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    error_message = None

    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        # Hata kontrolü: Eğer form boşsa, hata mesajı göster
        if not ticker or not start_date or not end_date:
            error_message = "Please fill all fields (Ticker, Start Date, End Date)."

        else:

            if not os.path.exists(model_path):
                # Veriyi indir
                stockPricePredictor.download_data(ticker, start_date, end_date)
                # Veriyi ölçeklendir
                stockPricePredictor.preprocess_data()
                # Dataset oluştur
                X,y = stockPricePredictor.create_dataset()
                # Build Model
                stockPricePredictor.build_model(input_shape=(X.shape[1],1))
                # Train model
                stockPricePredictor.train_model(X, y, epochs=15, batch_size=32)
                # Save model
                stockPricePredictor.model.save(model_path)
            else:
                stockPricePredictor.load_model(model_path)

            stockPricePredictor.download_data(ticker, start_date, end_date)
            stockPricePredictor.preprocess_data()
            X, y = stockPricePredictor.create_dataset()
            predicted_stock_price = stockPricePredictor.predict(X)
            predicted_stock_price = stockPricePredictor.inverse_transform(predicted_stock_price)
            actual_stock_price = stockPricePredictor.inverse_transform(y.reshape(-1, 1))

            # Grafik oluşturma ve kaydetme
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(actual_stock_price, color='blue', label='Actual Stock Price')
            ax.plot(predicted_stock_price, color='red', label='Predicted Stock Price')
            ax.set_title(f'{ticker} Stock Price Prediction')
            ax.set_xlabel('Time')
            ax.set_ylabel('Stock Price (USD)')
            ax.legend()

            # Grafiği bir BytesIO nesnesine kaydetme
            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)

            # Base64 formatına dönüştürme
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
            plt.close()  # Sonraki grafikleri engellemek için kapatıyoruz

    return render_template('index.html', plot_url=plot_url, error_message=error_message)
