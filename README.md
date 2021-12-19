# EuroMillionsPredictionAPI
Jesse Dingley, Damien Lalanne, Victor Maillot

### Installation

1. Create a virtual environment and install the required packages (`$ pip install -r requirements.txt`)
2. Go to the `/app` directory
3. execute the command `$ uvicorn api:app --reload`
4. In a browser open `http://localhost:8000/docs`
5. Try out the API.

### Usage:
- Predict the probability of a combination being a win (**POST** `/api/predict`).
- Generate a combination with a high probability of winning (**GET** `/api/predict`).
- Get model information such as the algorithm and training hyperparameters (**GET** `/api/model`).
- Add data to the dataset (**PUT** `/api/model`).
- Retrain the model with the new data added (**POST** `api/model/retrain`).

### Technical choices

- We have implemented a **random forest** model to make predictions because it's a small and efficient model. Another reason we chose this model is because we have **qualitative features** in the data (a draw is a sequence of random numbers that don't represent quantitative data).

- Our app is split into three distinct files:
    - `app/model.py`: module for training and testing Euromillions prediction model.
    - `app/preprocess_data.py`: module for preprocessing raw csv data to be able to use it for training.
    - `app/api.py`: API module.
    - 
- We use **pickle** to serialize and save our model so we don't have to retrain it at each launch. The model can be found in `rf_model.pickle`.

- Data is stored in two files:
   - `data/original/EuroMillions_numbers.csv`
   - `data/updated/updated_EuroMillions_numbers.csv`

   In `/original` we keep the original .csv for reference just in case.
   In  `/updated` we keep a copy of the original .csv which is updated  every time new data is added.
   
### Note

Due to the random nature of EuroMillions and so the data we have as well, our model will never be able to learn concrete patterns from the data. So predictions will be more or less random. So the more data we add to the model, the even harder it is to learn patterns from the data. This reflects the reality of this game (billions of possible draw combinations). 
