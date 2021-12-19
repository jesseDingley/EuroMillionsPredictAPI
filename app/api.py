"""API module.

This module contains code for defining the FastAPI api.
The entirety of the api code can be found here:
    - API definiton.
    - Combination class definition.
    - Draw class definition.
    - api request functions.
    
What you can do with this API:
    - Predict the probability of a combination being a win.
    - Generate a combination with a high probability of winning.
    - Get model information such as the algorithm and training hyperparameters.
    - Add data to the dataset.
    - Retrain the model with the new data added.
"""




"""Imports."""
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import model 
import preprocess_data as pp
import pickle
import datetime




"""API description."""
description = """

### A FastAPI app to make EuroMillions predictions.

### What you can do with this API:
- Predict the probability of a combination being a win (**POST** `/api/predict`).

- Generate a combination with a high probability of winning (**GET** `/api/predict`).

- Get model information such as the algorithm and training hyperparameters (**GET** `/api/model`).

- Add data to the dataset (**PUT** `/api/model`).

- Retrain the model with the new data added (**POST** `api/model/retrain`).
"""




"""Define API."""
app = FastAPI(
    title = "EuroMillionsPredictionAPI",
    description = description,
    contact={
        "name": "Jesse Dingley | Damien Lalanne | Victor Maillot",
        "email": "jesse.dingley@gmail.com"
    }
)




"""Constants."""
UPDATED_DATA_PATH = "../data/updated/updated_EuroMillions_numbers.csv"




"""Get model."""
with open("rf_model.pickle", "rb") as f:
    random_forest_model = pickle.load(f)




class Combination(BaseModel):
    """Represents a combination (sequence of digits).

    Attributes:
        N1 (int): first 'N number' of combination (1 <= N1 <= 50)
        N2 (int): second 'N number' of combination (1 <= N2 <= 50)
        N3 (int): third 'N number' of combination (1 <= N3 <= 50)
        N4 (int): fourth 'N number' of combination (1 <= N4 <= 50)
        N5 (int): fifth 'N number' of combination (1 <= N5 <= 50)
        E1 (int): first 'E number' of combination (1 <= E1 <= 12)  [BONUS NUMBER]     
        E2 (int): second 'E number' of combination (1 <= E2 <= 12)  [BONUS NUMBER]
    """
    N1: int = Field(default=1, ge=1, le=50, description="'Numéro' 1 need to be between 1 and 50", title="The first number of the combinaison")
    N2: int = Field(default=2, ge=1, le=50, description="'Numéro' 2 need to be between 1 and 50", title="The second number of the combinaison")
    N3: int = Field(default=3, ge=1, le=50, description="'Numéro' 3 need to be between 1 and 50", title="The third number of the combinaison")
    N4: int = Field(default=4, ge=1, le=50, description="'Numéro' 4 need to be between 1 and 50", title="The fourth number of the combinaison")
    N5: int = Field(default=5, ge=1, le=50, description="'Numéro' 5 need to be between 1 and 50", title="The fifth number of the combinaison")
    E1: int = Field(default=1, ge=1, le=12, description="'Etoile' 1 need to be between 1 and 12", title="The first bonus number of the combinaison")
    E2: int = Field(default=2, ge=1, le=12, description="'Etoile' 2 need to be between 1 and 12", title="The second bonus number of the combinaison")




class Tirage(Combination):
    """Represents a draw.

    Inherits the Combination class.

    Attributes:
        Date (datetime.date): date in format (yyyy-mm-dd).
        Winner (int): number of winners for given draw.
        Gain (int): amount of money won.
    """
    Date: datetime.date
    Winner: int = Field(default=0, ge=0, description="Winner need to be at least 0", title="Number of winners for given combination.")
    Gain: int = Field(default=100_000, ge=100_000, le=250_000_000, description="Gain need to be between 100.000 and 250.000.000", title = "Amount of money won.")




def model_to_dataframe(draw: Tirage):
    """Converts Tirage object to dataframe.

    Args:
        draw (Tirage): A draw.

    Returns:
        pandas.core.frame.DataFrame: converted draw object.
    """
    newDataframe = pd.DataFrame(data= {'Date': [draw.Date],
                                        'N1': [draw.N1],
                                        'N2': [draw.N2],
                                        'N3': [draw.N3],
                                        'N4': [draw.N4],
                                        'N5': [draw.N5],
                                        'E1': [draw.E1],
                                        'E2': [draw.E2],
                                        'Winner': [draw.Winner],
                                        'Gain': [draw.Gain]})
    return newDataframe




@app.post("/api/predict", tags = ["POST"])
async def predict_combination(combination: Combination):
    """Predicts the probability of a combination being a win.

    Returns two probabilities: 
    - the probability of the combination winning.
    - the probability of the combination not winning.

    Args:
        combination (api.Combination): the combination to make a prediction on.
    
    Returns:
        dict: Dictionary containing the two probabilities (where values are between 0 and 1).
    """
    probability = model.predict_combination(random_forest_model, combination.dict())
    return {"win probability": probability, 
            "lose probability": 1-probability}




@app.get("/api/predict", tags = ["GET"])
async def generate_probable_combination():
    """Generates a combination with a high probability of winning.

    Returns:
        dict: Combination dictionary (where keys are "N1", "N2", ... and values are combination numbers). 
    """
    return model.get_probable_combination(random_forest_model)




@app.get("/api/model", tags = ["GET"])
async def get_model_information():
    """Gets model information.

    Returns a dictionary containing model information such as:
        - Performance metric.
        - Algorithm used.
        - Training hyperparameters.

    Returns:
        dict
    """
    return {"peformance metric": model.PERFORMANCE_METRIC,
            "algorithm name": model.ALGORITHM,
            "training parameters": {
                "number of trees": model.NUM_TREES,
                "test size": model.TEST_SIZE
                }
            }




@app.put("/api/model", tags = ["PUT"])
async def add_data_to_model(added: Tirage):
    """Adds new row to the dataset.

    Updates dataset csv file which is now the concatenation of the current dataset and the new row (a draw).

    Args:
        added (Tirage): new draw to add.

    Returns:
        dict: Validation message.
    """
    baseDataframe = pp.create_df(UPDATED_DATA_PATH)
    addDataframe = model_to_dataframe(added)
    baseDataframe = baseDataframe.append(addDataframe, ignore_index=True).reset_index(drop=True)
    baseDataframe.to_csv(UPDATED_DATA_PATH, sep=';', index = False)
    return {"Validation message": "Your data has been correctly added to the dataset"}




@app.post("/api/model/retrain", tags = ["POST"])
async def retrain_model():
    """Retrains a new model.

    Retrains the random forest model and saves it to a pickle file for later usage.

    Returns:
        dict: validation message.
    """
    new_random_forest_model = model.train_from_source(UPDATED_DATA_PATH)
    with open("rf_model.pickle", "wb") as f:
        pickle.dump(new_random_forest_model, f)
    return {"Validation message": "Model successfully retrained."}