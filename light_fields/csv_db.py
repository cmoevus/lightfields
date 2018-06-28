import numpy as np
import pandas as pd

from light_fields.spectral_distribution import SpectralDistribution

class CSVDatabase(object):
    """Build a database-like structure from a CSV file.

    Parameters:
    -----------
    db_file: str or pd.DataFrame
        Path to the csv file containing the database, or pandas DataFrame of the database

    """

    def __init__(self, db_file=None):
        """Instanciate a CSV database."""
        if type(db_file) != pd.DataFrame:
            self.db_path = db_file
            if db_file is not None:
                self.load_db()
        else:
            self.db = db_file
            self.db_path = None

    def load_db(self, db_file=None):
        """Load a CSV file as its database."""
        if db_file is None:
            db_file = self.db_path
        self.db = pd.read_csv(db_file)

    def find(self, **kwargs):
        """Return a list of rows, from the database, that fit the given criteria.

        All given criteria have to be satisfied for a row to be returned.
        """
        expression = True
        for k, i in kwargs.items():
            ref = self.db[k]
            if type(i) == str:
                ref = ref.str.lower()
                i = i.lower()
            expression = np.logical_and(ref == i, expression)
        return self.db.where(expression).dropna(how='all').reset_index()

    def add(self, **kwargs):
        """Add a component to the database."""
        i = len(self.db)
        for k, v in kwargs.items():
            self.db.loc[i, k] = v
        self.db.to_csv(self.db_path, index=False)

    def __iter__(self):
        """Make the database iterable."""
        self.counter = 0
        return self

    def __next__(self):
        """Return next row in the database."""
        self.counter += 1
        try:
            return self.db.iloc[self.counter - 1]
        except IndexError:
            raise StopIteration

    def __len__(self):
        """Count the number of rows of the database."""
        return len(self.db)
