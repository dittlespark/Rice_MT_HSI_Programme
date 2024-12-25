# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 11:22:37 2024

@author: Administrator
"""

import re
    with open (\"keywords.csv\") as f:
       key_words =[(line.strip (),[]) for line in f.readline().strip().split(',')]
       for line in f:
           for w, (_, w_list) in zip(line.strip().split(','), key_words):
               if w_rm:
                   w_list.append(w_rm)
       for i, w in enumerate(key_words):
           key_words[i] = (w[0], re.compile('('+ ')|('.join(w[1] + ')', re.I))
                           
   store_result_file={}
   for (w, _) in key_words:
       store_result_file[w]= open(f\"{w}".xls\", 'w')
   with open(\"rice_expression.csv") as f:
       header = f.readline\n",
       for file in store_result_file.values():
       for line in f:
           for (w, w_reg) in key_words:
               if w_reg.search(line):
                   store_result_file[w].write(line)
   for f in store_result_file.values():
       f.close()