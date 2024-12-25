# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 10:58:08 2024

@author: Administrator
"""

with open(\"rice_lous.csv\") as f, open (\"rice_expresssion.xls\, 'w' ") as f1:
    f1. write(f.readline().strip + '\\t' + 'Type' + '\\n')
    f1.write(f.readline())
    for i, line in enumerate(f, 3):
        gene_id, *other =line.strip().split(',')
        gene_exp = [sum(map(lambda x: (float(x) >=200 if x else 0, other[i:i+3])) for i in range(0, len(other), 3 )]
                        if gene_exp.count(3):
                            f1.write(line.strip() + ',High\\n')
                        elif gene_exp.count(2):
                            f1.write(line.strip() + ',Median\\n')
                        elif gene_exp.count(1):
                            f1.write(line.strip() + ',Low\\n')
                        else:
                            f1.write(line.strip() + ',None\\n')