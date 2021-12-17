"""A module for preprocessing raw csv data.

This module contains four functions for preparing raw csv data for usage.

    Typical usage example:

    raw_df = create_df("../data/EuroMillions_numbers.csv")
    combination = generate_random_combination()
    new_df = add_data(raw_df, 10)
    add_binary_winner_column(new_df)
    df = preprocess("../data/EuroMillions_numbers.csv", 10)
    comb_dict = combination_array_to_dict(combination)
"""


from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import model 
import preprocess_data as pp

"""Initalize model."""
DATA_PATH = "../data/EuroMillions_numbers.csv"
UPDATED_DATA_PATH = "../data/updated_EuroMillions_numbers.csv"
random_forest_model = model.train_from_source(DATA_PATH)

app = FastAPI()

class Combination(BaseModel):
    N1: int = Field(default=1, ge=1, le=50, description="'Numéro' 1 need to be between 1 and 50", title="The first number of the combinaison")
    N2: int = Field(default=2, ge=1, le=50, description="'Numéro' 2 need to be between 1 and 50", title="The second number of the combinaison")
    N3: int = Field(default=3, ge=1, le=50, description="'Numéro' 3 need to be between 1 and 50", title="The third number of the combinaison")
    N4: int = Field(default=4, ge=1, le=50, description="'Numéro' 4 need to be between 1 and 50", title="The fourth number of the combinaison")
    N5: int = Field(default=5, ge=1, le=50, description="'Numéro' 5 need to be between 1 and 50", title="The fifth number of the combinaison")
    E1: int = Field(default=1, ge=1, le=12, description="'Etoile' 1 need to be between 1 and 12", title="The first bonus number of the combinaison")
    E2: int = Field(default=2, ge=1, le=12, description="'Etoile' 2 need to be between 1 and 12", title="The second bonus number of the combinaison")

class Tirage(BaseModel):
    Date: int # à changer pour mettre le type datetime
    N1: int = Field(default=1, ge=1, le=50, description="'Numéro' 1 need to be between 1 and 50", title="The first number of the combinaison")
    N2: int = Field(default=2, ge=1, le=50, description="'Numéro' 2 need to be between 1 and 50", title="The second number of the combinaison")
    N3: int = Field(default=3, ge=1, le=50, description="'Numéro' 3 need to be between 1 and 50", title="The third number of the combinaison")
    N4: int = Field(default=4, ge=1, le=50, description="'Numéro' 4 need to be between 1 and 50", title="The fourth number of the combinaison")
    N5: int = Field(default=5, ge=1, le=50, description="'Numéro' 5 need to be between 1 and 50", title="The fifth number of the combinaison")
    E1: int = Field(default=1, ge=1, le=12, description="'Etoile' 1 need to be between 1 and 12", title="The first bonus number of the combinaison")
    E2: int = Field(default=2, ge=1, le=12, description="'Etoile' 2 need to be between 1 and 12", title="The second bonus number of the combinaison")
    Winner: int = Field(default=0, ge=0, le=1, description="Winner need to be between 0 and 1", title="The combinaison has a winner or not")
    Gain: int = Field(default=100_000, ge=100_000, le=230_000_000, description="Gain need to be between 100.000 and 230.000.000")

def model_to_dataframe(model: Tirage):
    newDataframe = pd.DataFrame(data= {'Date': [model.Date],
                                        'N1': [model.N1],
                                        'N2': [model.N2],
                                        'N3': [model.N3],
                                        'N4': [model.N4],
                                        'N5': [model.N5],
                                        'E1': [model.E1],
                                        'E2': [model.E2],
                                        'Winner': [model.Winner],
                                        'Gain': [model.Gain]})
    return newDataframe




@app.post("/api/predict")
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
    return {"proba gain": probability, 
            "proba perte": 1-probability}




@app.get("/api/predict")
async def generate_probable_combination():
    """Generates a combination with a high probability of winning.

    Returns:
        dict: Combination dictionary (where keys are "N1", "N2", ... and values are combination numbers). 
    """
    return model.get_probable_combination(random_forest_model)

@app.get("/api/model")
async def get_model_information():
    return {"metrique performance": model.PERFORMANCE_METRIC,
            "nom algorithme": model.ALGORITHM,
            "parametres entrainement": {
                "nombre arbres": model.NUM_TREES,
                "taille base test": model.TEST_SIZE
                }
            }


@app.put("/api/model")
async def add_data_to_model(added: Tirage):
    """Allow the user to add data in the dataset

        Input: object of type Tirage (automaticaly transforms your json input into a object of type Tirage) with shape :
            `Date: int
            N1: int
            N2: int
            N3: int
            N4: int
            N5: int
            E1: int
            E2: int
            Winner: int
            Gain: int`

        Returns:
            json with shape :
            `{"Validation message": <string>}`

    """

    # if ((added.N1 < 1 or added.N1 > 50) or (added.N2 < 1 or added.N2 > 50) or (added.N3 < 1 or added.N3 > 50) or (added.N4 < 1 or added.N4 > 50) or (added.N5 < 1 or added.N5 > 50) or (added.E1 < 1 or added.E1 > 12) or (added.E2 < 1 or added.E2 > 12)):
    #     return {"Error message": "One of your number does not suit the requierement : N1 to N5 between 1 and 50, E1 et E2 between 1 and 12"}

    baseDataframe = pp.create_df('../data/EuroMillions_numbers.csv')
    addDataframe = model_to_dataframe(added)
    baseDataframe = baseDataframe.append(addDataframe).reset_index(drop=True)
    baseDataframe.to_csv(UPDATED_DATA_PATH, sep=';')

    return {"Validation message": "Your data has been correctly added to the dataset"}

@app.post("/api/model/retrain")
async def retrain_model():
    global random_forest_model
    random_forest_model = model.train_from_source(UPDATED_DATA_PATH)
    return {"Validation message": "Model successfully retrained."}

# J'ai définit l'api avec "app = FastAPI()" et mon fichier s'appelle "api.py". Donc pour lancer uvicorn il faut utiliser la commande terminale :
#   > uvicorn api:app --reload (le reload permet de refresh les changements de manière dynamique)
# Une fois le serveur uvicorn lancé, il faut aller sur le localhost et rajouter dans l'url /docs
# Ca permet de voir toute la doc générée grâce aux commentaires des fonctions et également de tester les méthodes selon les requêtes d'api définies.
# Utiliser le bouton "Try it out" pour vérifier que la méthode fonctionne bien. Cela fourni aussi la commande utilisée pour requêter l'api (la commande curl)
# Il suffit de copier cette commande dans un 2nd terminal (le premier héberge le serveur uvicorn) et vérifier que tout marche bien :)