from model import *
import pandas as pd

def Assignment_1():
  data: list[list[int]] = [[2, 4], [4, 6], [4, 2], [6, 4], [6, 6]]
  label = ['Red', 'Red', 'Blue', 'Blue', 'Red']
  d_new = [5, 5]
  centroid = Centroids_2D(data, label)
  print(centroid.predict(d_new))

def Assignment_2():
  df = pd.read_csv('Data/data.csv')
  data, label = df.iloc[:, 0:4], df.iloc[:, 4] 
  model = ID3().fit(data, label)

if __name__ == "__main__":
  Assignment_2()