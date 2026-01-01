"""
Converts MNIST dataset to sqlite database
"""

import sqlite3
import io
import numpy as np
import torchvision.datasets as datasets
from tqdm import tqdm

DB_NAME = "data/mnist.db"
BATCH_SIZE = 1000

def init_db(db_path):
    conn = sqlite3.connect(db_path) # active connection object
    cursor = conn.cursor()
    # images are Binary Large Objects (BLOB), so they are stored as bytes
    # labels are stored as integers - the answer for the MNIST problem
    # split stores if the column belongs to test or train
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mnist_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image BLOB NOT NULL, 
            label INTEGER NOT NULL,
            split TEXT NOT NULL
        )
    ''')
    #Creates a database index on the split and label columns for fast search
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_split_label ON mnist_data (split, label)')
    conn.commit() # save changes of table and index to the file
    return conn # return object to start inserting data later

def prepare_data(dataset, split_name):
    #img from dataset is PIL.Image.Image
    for img, label in dataset:
        img_array = np.array(img, dtype=np.uint8) #convert PIL to numpy array
        #extract bytes from numpy array
        with io.BytesIO() as bio:
            np.save(bio, img_array)
            img_bytes = bio.getvalue()
        #stream data into sqlite database to save_to_sqlite
        yield (img_bytes, label, split_name)

def save_to_sqlite(db_path):
    print("⬇️  Downloading MNIST dataset...")
    # Transformations are not needed for storage, we store raw uint8
    train_set = datasets.MNIST(root='./data', train=True, download=True)
    test_set = datasets.MNIST(root='./data', train=False, download=True)
    
    conn = init_db(db_path)
    cursor = conn.cursor()
    
    for split_name, dataset in [('train', train_set), ('test', test_set)]:
        print(f"📦 Inserting {split_name} samples...")
        data_iter = prepare_data(dataset, split_name)
        chunk = []
        for item in tqdm(data_iter, total=len(dataset)):
            chunk.append(item)
            if len(chunk) >= BATCH_SIZE:
                cursor.executemany('INSERT INTO mnist_data (image, label, split) VALUES (?, ?, ?)', chunk)
                conn.commit()
                chunk = []
        if chunk:
            cursor.executemany('INSERT INTO mnist_data (image, label, split) VALUES (?, ?, ?)', chunk)
            conn.commit()

    print(f"✅ Data saved to {db_path}")
    conn.close()

if __name__ == "__main__":
    save_to_sqlite(DB_NAME)