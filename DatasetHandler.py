import csv
import numpy as np
from sklearn.preprocessing import StandardScaler

class Dataset:
    def __init__(self, filename, modes):
        self.modes = modes
        self.filename = filename
        print('Dataset handler is ready using ', filename)
    
    def WriteRow(self, row):
        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
    
    def ReadCSV(self, transform):
        try:
            x, y = [], []
            corrupted = 0
            with open(self.filename, 'r') as file:
                csvreader = csv.reader(file)
                for row in csvreader:
                    if len(row) == 15:
                        x.append(row[0 : 14])
                        y.append(row[14])
                    else:
                        corrupted += 1
        
            x, y = x[1:], y[1:]
            if transform:
                scaler = StandardScaler()
                x = scaler.fit_transform(x)
            
            classes = list(set(y))
            y = [classes.index(i) for i in y]
            
            if self.modes['Info']:
                print('Total Classes Detected', classes)
                print('Corrupted entries found:', corrupted)
                
            return np.array(x, dtype='int'), np.array(y, dtype='int'), classes
        except Exception as e:
            if self.modes['Warning']:
                print(e)
            
            return False