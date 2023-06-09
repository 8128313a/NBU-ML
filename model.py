import pickle
import re
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)


classes = [
 0,
 1
]

def get_info():
    print("The person has the following income:")
    income = float(input())
    print("The person has the following lot size")
    lot_size = float(input())
    new_X = np.array((income,lot_size)).reshape(1,-1)
    prediction = model.predict(new_X)
    return prediction
    