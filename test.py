`# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 19:44:15 2020

@author: Administrator
"""
import torch
import torchvision
import threading
import time

# s = 'cadeda'
# if len(s) == 0:
#     print ('')
# res = s[0]
# for i in range(len(s)):
#     print('i当前：%d' %i)
#     subs = s[i]
#     j = 1
#     while i-j>=0 and i+j < len(s) and s[i-j] == s[i+j]:
#         subs = s[i-j]+subs+s[i+j] 
#         j += 1
#     if len(subs) > len(res):
#         res = subs 
# print(res)

import threading
def pr(stre):
    for i in range(10):
        time.sleep(1)
        print(stre)
    
    
    
t1 = threading.Thread(target= pr,args=(1,))
t2 = threading.Thread(target= pr,args=(2,))
t1.start()
t2.start()
 
    