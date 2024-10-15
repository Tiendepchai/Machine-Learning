from queue import PriorityQueue
from collections import defaultdict
from math import *
import pandas as pd
import numpy as np
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

class ID3:
  def __init__(self) -> None:
    self.data = None
    self.label = None
    self.entropy = 0
    self.features = None
    self.IGs = list()
    self.greatest_IG = None

  def calculate_entropy(self, data, label:pd.DataFrame):
    total_rows = len(data)
    count = label.value_counts()
    target_values = label.values
    target_values = np.unique(target_values)

    entropy = 0
    for value in target_values: 
      pi = count[value]
      proportion = pi / total_rows
      entropy += (- proportion * log2(proportion))
    return entropy

  def __cal_IG(self):
    for i in self.features:
      self.IGs.append((self.entropy - self.features[i], i))
    self.IGs.sort()
    self.greatest_IG = self.IGs[-1]

  def cal_H_features(self, data:pd.DataFrame, label:pd.DataFrame):
    title = data.columns.values
    ans = {}
    lbl = label.values
    target_values = np.unique(label.values)
    for i in title:
      sett = np.unique(data[i].values)
      cnt_vals = data[i].value_counts()
      arr = data[i].to_list()
      l = len(arr)
      IG = 0
      for x in sett:
        cc = 0
        cnt = {x: 0 for x in target_values}
        for y in range(l):
          if arr[y] == x:
            cnt[lbl[y]] += 1
            cc += 1
        # Calculate entropy
        
        entropy = 0
        for value in cnt: 
          pi = cnt[value]
          if pi != 0:
            proportion = pi / cc
            entropy += (- proportion * log2(proportion))
          
        IG += (cnt_vals[x]/l)*entropy
      ans[i] = IG
    return ans

  def fit(self, data, label):
    self.data = data
    self.label = label
    self.entropy = self.calculate_entropy(data, label)
    self.features = self.cal_H_features(data, label)
    self.__cal_IG()
    print(self.greatest_IG)