# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 09:01:15 2020

@author: lenovo
"""

import Bio
from Bio import SeqIO
'''import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)'''
gene_ids_temp = []
X = []
y = []
for seq_record in SeqIO.parse("Cytoplasm_indep.fasta", "fasta"):
    gene_ids_temp.append(seq_record.id)
    X.append(seq_record.seq)
    y.append("Cystoplasm")
 
gene_ids = []
for ids in gene_ids_temp:
    gene_ids.append((str(ids).split("#")[1]))
                     
for item in gene_ids:
    print(item)