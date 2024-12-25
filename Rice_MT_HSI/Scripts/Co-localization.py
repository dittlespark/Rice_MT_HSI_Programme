# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 10:33:29 2024

@author: Administrator
"""

from collections import defaultdict
def parse_info(file):
    with open(file) as f:
        store_info =defaultdict(list)
        header= '\\t'.join(filter(bool, f.readline().strip().split(',')))
        for line in f:
            content =list(filter(bool, line.strip().split(',')))
            store_info[content[0]].append((float(content[2], '\\t'.join(content))))
    return store_info, header

store_dai_xie, header1= parse_info(\"MT_snp.csv\")
store_guang_pu, header2 = parse_info(\"HP_snp.csv\")
     with open (\"MT_HP_colocalization.xls\", 'w') as f
        f.write(header1 + '\\t' + header2 + '\\n')
        for (t, v) in store_dai_xie.items():
            g = store_guang_pu[t]
            if g:
                offset_i=0
                for v_v, v_line in v:
                    record_flag=1
                    for i, (g_v, g_line) in enumerate(g[offset_i:]):
                        if abs(g_v -v_v) <=0.3:
                            if record_flag:
                                offset_i += i
                                record_flag =0
                            f.write(v_line + '\\t' + g_line + '\\n')
                        else:
                            break
                            
                                    