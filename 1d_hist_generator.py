# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 07:36:11 2018

@author: jasplund
"""

import math
import numpy as np
import copy
import random
import matplotlib.pyplot as plt # only used for the histogram at the end

#plotly.tools.set_credentials_file(username='cecilartavion', api_key='ADD KEY HERE')

#Use this distribution for when there are 10 precincts
dist = [1,0,0,0,1,1,0,0,0,1] 

##Use this distribution when there are 100 precincts.
#dist = np.zeros(100)
#precincts = [2,3,4,5,6,7,19,20,22,23,24,28,29,31,32,35,36,37,40,41,47,49,50,54,59,60,62,63,64,66,70,72,79,80,84,85,89,93,98,99]
#for precinct in precincts:
#    dist[precinct] = 1
    
num_yellow_seats=[]
for i1 in range(len(dist)-1):
    for i2 in range(i1+1,len(dist)-1):
        for i3 in range(i2+1,len(dist)-1):
            for i4 in range(i3+1,len(dist)-1):
                temp_yellow_count=0
                dist1 = dist[:i1+1]
                dist2 = dist[i1+1:i2+1]
                dist3 = dist[i2+1:i3+1]
                dist4 = dist[i3+1:i4+1]
                dist5 = dist[i4+1:]
                if np.sum(dist1)>len(dist1)/2:
                    temp_yellow_count += 1
                if np.sum(dist1)==len(dist1)/2:
                    temp_yellow_count += 0.5
                if np.sum(dist2)>len(dist2)/2:
                    temp_yellow_count += 1
                if np.sum(dist2)==len(dist2)/2:
                    temp_yellow_count += 0.5
                if np.sum(dist3)>len(dist3)/2:
                    temp_yellow_count += 1
                if np.sum(dist3)==len(dist3)/2:
                    temp_yellow_count += 0.5
                if np.sum(dist4)>len(dist4)/2:
                    temp_yellow_count += 1
                if np.sum(dist4)==len(dist4)/2:
                    temp_yellow_count += 0.5
                if np.sum(dist5)>len(dist5)/2:
                    temp_yellow_count += 1
                if np.sum(dist5)==len(dist5)/2:
                    temp_yellow_count += 0.5
                num_yellow_seats.append(temp_yellow_count)
                
bins = np.arange(0,5,0.5)-.175
width=0.7*(bins[1]-bins[0])
print(len(num_yellow_seats))
plt.hist(num_yellow_seats,bins=bins,width=width)
plt.ylabel('Frequency')
plt.xlabel('Number of districts (out of 5) for Yellow party')
#plt.savefig('1d_hist_gy.eps', format='eps', dpi=1000)
plt.show()
