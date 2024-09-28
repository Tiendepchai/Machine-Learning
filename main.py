from model import *

def cal_distance(A: list[int], B: list[int]):
  return ((A[0]-B[0])**2 + (A[1]-B[1])**2)**(1/2)

if __name__  == "__main__":
  data: list[list[int]] = [[2, 4], [4, 6], [4, 2], [6, 4], [6, 6]]
  label = ['Red', 'Red', 'Blue', 'Blue', 'Red']
  d_new = [5, 5]
  centroid = Centroids_2D(data, label)
  print(centroid.predict(d_new))

  