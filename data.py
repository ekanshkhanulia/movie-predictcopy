from io import StringIO
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch

class MovieLensDataset:
    def __init__(self, movies_file, ratings_file, maxlen=50):
        # Load the data
        self.movies = self.read_data(movies_file, ['MovieID', 'Title', 'Genres'], encoding='latin-1')
        self.ratings = self.read_data(ratings_file, ['UserID', 'MovieID', 'Rating', 'Timestamp'])

        # Data Preprocessing
        self.preprocess()

        # Leave-one-out split
        self.train_X = []
        self.val_X = []
        self.test_X = []
        self.train_y = []
        self.val_y = []
        self.test_y = []
        self.user_histories = [] # for negative sampling during training

        self.train_val_test_split(maxlen)
        
        self.splits = {
            'train': (self.train_X, self.train_y), 
            'val': (self.val_X, self.val_y), 
            'test': (self.test_X, self.test_y)
        }
        

    def read_data(self, filename, names, encoding='utf-8'):
        """ Allows for reading the data faster."""
        # Replace '::' with a single char separator so we can use C engine instead of Python engine
        # (C engine can read only one character so for the two '::' we would need Python)
        with open(filename, 'r', encoding=encoding) as f:
            content = f.read().replace('::', '\t')
        return pd.read_csv(StringIO(content), sep='\t', names=names)

    def preprocess(self):
        """Performs data preprocessing"""
        # Since some MovieIDs do not correspond to a movie (there are gaps in the IDs, e.g. 1, 2, 4) due to accidental duplicate entries and/or test entries,
        # we re-index the movie id's to ensure they all correspond to a movie
        movie_encoder = {m: i+1 for i, m in enumerate(self.movies['MovieID'])}
        self.movies['MovieID'] = self.movies['MovieID'].map(movie_encoder)
        self.ratings['MovieID'] = self.ratings['MovieID'].map(movie_encoder)

        # Convert the explicit ratings to binary feedback
        # if rating >= 4 then label 1 (positive)
        self.ratings['Rating'] = (self.ratings['Rating'] >= 4).astype(int)
        # keep only positives (drop the movies rated <4)
        self.ratings = self.ratings[self.ratings['Rating'] == 1]
        self.ratings.drop('Rating', axis=1, inplace=True)

        # Generate chronological interaction sequences for each user
        # Order by timestamp
        self.ratings = self.ratings.sort_values(by='Timestamp', ascending=True)

        # Filter out users with fewer than 5 interactions
        # Replace the for loop with this:
        user_counts = self.ratings.groupby('UserID')['MovieID'].count()
        valid_users = user_counts[user_counts >= 5].index
        self.ratings = self.ratings[self.ratings['UserID'].isin(valid_users)]

    def train_val_test_split(self, maxlen):
        """
        Leave-one-out split:
        - train/val input: all but the last two items
        - test input:      all but the last item (includes second-to-last)
        """
        for _, group in self.ratings.groupby('UserID'):
            movies = list(group['MovieID'])
            train_val_seq = movies[:-2]
            test_seq = movies[:-1]

            def pad(seq):
                # If sequence is longer than maxlen, keep only the most recent
                seq = seq[-maxlen:]
                # If sequence is shorter than maxlen, left-pad with zeros
                return [0] * (maxlen - len(seq)) + seq

            # Inputs
            self.train_X.append(pad(train_val_seq))
            self.val_X.append(pad(train_val_seq))
            self.test_X.append(pad(test_seq))

            # Targets
            # For training, the targets are the sequences shifted one step to the right
            train_target = movies[1:-1] 
            self.train_y.append(pad(train_target + [movies[-2]]))
            # For validation and test, they are the second to last or last item in sequence, respectively
            self.val_y.append(movies[-2])  
            self.test_y.append(movies[-1])

            # Keeping track of all movies seen by the user, to use later for negative sampling
            self.user_histories.append(set(movies))
        
    def get_loader(self, split, batch_size):
        """Converts a data split into a PyTorch DataLoader for batch sampling during training."""
        X = torch.tensor(self.splits[split][0], dtype=torch.int64) #seq
        y = torch.tensor(self.splits[split][1], dtype=torch.int64)

        # Only shuffles training data
        return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=(split == 'train'))
    
