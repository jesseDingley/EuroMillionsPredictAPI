# EuroMillionsPredictionAPI

### Installation

1. Create a virtual environment and install the required packages (`pip install -r requirements.txt`)
2. Go to the `EuroMillionsPredictionAPI/app` directory
3. execute `uvicorn api:app --reload`
4. In a browser open `http://localhost:8000/docs`
5. Try out the API.

### Technical choices

- We have implemented a random forest to make predictions because it's a small model and the data has qualitative features.
- /arguments
