import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset:
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data = None
        self.X = None
        self.y = None
    
    def load(self) -> pd.DataFrame:
        self.data = pd.read_csv(self.csv_path)
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]
        return self.data

    def split(self, test_size: float = 0.2, random_state: int = 42):
        if self.data is None:
            raise ValueError("No data loaded.")
        return train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )