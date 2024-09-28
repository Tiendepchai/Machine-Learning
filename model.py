from queue import PriorityQueue
from collections import defaultdict
class KNN_2D:
  def __init__(self, data: list[list[int]], label: list[any]) -> None:
    self.data = data
    self.label = label

  def cal_distance(self, A: list[int], B: list[int]):
    return ((A[0]-B[0])**2 + (A[1]-B[1])**2)**(1/2)
  
  def show(self, B: list[int]):
    for i in range(len(self.data)):
      print(self.data[i], self.label[i])
    
  def predict(self, B: list[int], K: int):
    arr = []
    for i in range(len(self.data)):
      arr.append((self.cal_distance(self.data[i], B), self.label[i]))
    arr.sort()
    
    ans = defaultdict(int)
    for i in range(K):
      ans[arr[i][1]] += 1

    most_label = None
    m = 0
    for i in ans:
      if ans[i] >= m:
        m = ans[i]
        most_label = i
    return most_label

class Centroids_2D:
  def __init__(self, data: list[list[int]], label: list[any]) -> None:
    self.data = defaultdict(list)
    for i in range(len(data)):
      self.data[label[i]].append(data[i])

  def cal_distance(self, A: list[int], B: list[int]):
    return ((A[0]-B[0])**2 + (A[1]-B[1])**2)**(1/2)
  
  def show(self):
    for i in self.data:
      print(i, self.data[i])
  
  def predict(self, B: list[int]):
    arr = {x:[0, 0] for x in self.data}
    ans = PriorityQueue()
    for i in arr:
      x, y = 0, 0
      cnt = 0
      for j in self.data[i]:
        x += j[0]
        y += j[1]
        cnt += 1
      arr[i] = [x/cnt, y/cnt]
    
    for i in arr:
      ans.put((self.cal_distance(arr[i], B), i))
    
    return ans.get()[1]

