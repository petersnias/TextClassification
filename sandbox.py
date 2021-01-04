#!/usr/bin/python

import pandas as pd
import random

D = pd.DataFrame()
A = pd.DataFrame(columns={'A','B','C'})

for i in range(5):
    A['A'] = ['0','1','2']
    A['B'] = ['2','4','6']
    A['C'] = ['3','5','7']
    a = [D,A]
    D = pd.concat(a)
