import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import datetime
import xlsxwriter as xw 

base_path = 'database/'

header = ["nose_x","nose_y","neck_x","neck_y","Rshoulder_x","Rshoulder_y","Relbow_x","Relbow_y",
          "Rwrist_x","RWrist_y","LShoulder_x","LShoulder_y","LElbow_x","LElbow_y","LWrist_x",
          "LWrist_y","RHip_x","RHip_y","RKnee_x","RKnee_y","RAnkle_x","RAnkle_y","LHip_x","LHip_y",
          "LKnee_x","LKnee_y","LAnkle_x","LAnkle_y","REye_x","REye_y","LEye_x","LEye_y","REar_x",
          "REar_y","LEar_x","Lear_y","class"]

file = open('sit_2.txt','r')  # file name that open to read (.txt)
file_read = file.read()

workbook = xw.Workbook('sit2.csv')      # file name that want to create and keep
worksheet = workbook.add_worksheet("sheet1")    # default
text1line = []
data = []
first = 0
last = 0
mess = ''

for i in range(0,len(file_read)):    
    if ' ' in file_read[i]:
        last = i
        r = last - first
        for j in range(0,r):
            mess = mess+file_read[first+j]
        text1line.append(mess)
        first = i+1
        print(mess)
        mess = ''
        
    if "\n" in file_read[i]:
        last = i
        r = last - first
        for j in range(0,r):
            mess = mess+file_read[first+j]
        text1line.append(mess)
        data.append(text1line)
        print(len(text1line))
        text1line = []
        first = i+1
        mess = ''
Ldata = len(data)


for k in range(0,len(header)):
    worksheet.write(0,k,header[k])
offset = 0

for row in range(offset,offset+Ldata):
    for col in range(0,36):
        worksheet.write(row+1,col,data[row-offset][col])
        print("row = ",row)
        print("col = ",col)
    worksheet.write(row+1,36,"sit") # define class (sit stand bla bla...)
    
workbook.close() 