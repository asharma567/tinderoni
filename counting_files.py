import sys
chrCounter = 0
import pandas as pd


lis=[]
f_names =[]
uniq=[]

for line in sys.stdin:
    f_names.append(line)
    time_stamp, name, num= line.split('_face.jpg')[0].split('_')
    lis.append(num)
    uniq.append(time_stamp + name)



# for num in range(len(lis)):
prev_num = lis[0]
another_lis = []
for idx, curr_num in enumerate(lis[1:]):
    if curr_num <= prev_num: 
        another_lis.append(f_names[idx])
    prev_num = curr_num

print ('\n'.join(map(str,another_lis)))