from app.models import PredictionRequest
from app.utils import get_model,transform_to_dataframe

model = None

def get_prediction(request: PredictionRequest) -> float:
    data_to_predict = transform_to_dataframe(request)
    prediction = model.predict(data_to_predict())[0]
    return max(0,prediction)