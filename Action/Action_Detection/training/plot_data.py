import pandas as pd 
import matplotlib.pyplot as plt
import csv

with open("data.csv", "rb") as f:
    reader = csv.reader(f)
    i = reader.next()
    print(i)
#plt.scatter(dataf['nose_x'])
#plt.show()

