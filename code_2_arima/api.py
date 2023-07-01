# Custom
from pull_data import read_data, update_data

# Model
from statsmodels.tsa.arima.model import ARIMA

# API
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from starlette import status

app = FastAPI()


update_data(12, 'data/raw_data.csv')
bom = read_data('data/raw_data.csv')


# Preprocessing
ws_cols = ['9am wind speed (km/h)', '3pm wind speed (km/h)']
bom[ws_cols] = bom[ws_cols].replace(['Calm'], 0).astype(float)


def check_models_trained():
    if not 'model' in globals():
        raise HTTPException(
            status_code=400, detail='Please train model first')


@app.post('/train_arima/', status_code=status.HTTP_204_NO_CONTENT)
def train_arima(target: str = Query(..., description="What feature should be predicted? e.g. Maximum temperature (°C) or Evaporation (mm)")):
    global model
    # Autoregressive Integrated Moving Average (ARIMA)
    model = ARIMA(bom[target], order=(range(1, 40), 1, range(1, 40)))

    # # Seasonal Autoregressive Integrated Moving Average (SARIMA)
    # return ARIMA(df[target], order=(10, 1, 10), seasonal_order=(2,2,2,12))

    model = model.fit()


@app.get('/forecast/')
def forecast(request):
    check_models_trained()
    if request.isdigit():
        request = int(request)

    prd = model.get_forecast(request)
    prd_df = prd.conf_int(alpha=0.05)

    return model.predict(
        start=prd_df.index[0], end=prd_df.index[-1])


if __name__ == "__main__":
    uvicorn.run('api:app', app_dir='code', reload=True)
