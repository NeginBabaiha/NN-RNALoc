# -*- coding: utf-8 -*-
"""
#Get gene info from rnalocate ( gene ENTREZ id, String id, organism)
Created on Tue Aug  4 08:44:21 2020

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 08:44:10 2020

@author: lenovo
"""

import pandas as pd
import os

import os

data = pd.read_excel('organismUnique.xlsx')
df = pd.DataFrame(data, columns= ['Entry','Status','Entry name','Cross-reference (GeneID)','Cross-reference (STRING)','Organism','Interacts with'])
arr = df.to_numpy()

'''count_human = 0
count_mouse = 0
count_other = 0
for item in arr:
    if( (item[5]) == "Homo sapiens (Human)" ):
        count_human=count_human+1
    if( (item[5]) == "Mus musculus (Mouse)" ):
        count_mouse=count_mouse+1
    else:
        count_other = count_other+1
        
print("Human",count_human)
print("Mouse",count_mouse)
print("Other",count_other)'''
#Get gene information including organism and gene ids and prot ids
f = open('gene_ids_unique.txt', 'r')
genes = [line.rstrip() for line in f]
f.close()
gene_info = []
temp = []
for gen in genes:
    temp.append(gen)
    for item in arr:
        #print(item[3].split(";")[0])
        if( (item[3].split(";")[0]) == gen ): #gene id
            temp.append( str(item[4]).split(";")[0] ) #prot id
            temp.append(item[5]) #Organism
            
    
    gene_info.append(temp)
    temp = []

print(len(gene_info))
for i in range(10):
    print(gene_info[i])
    
'''import pickle

with open("gene_info_rnalocate.txt", "wb") as fp:   #Pickling
    pickle.dump(gene_info, fp)

import sys
sys.exit()'''
'''with open("test.txt", "rb") as fp:   # Unpickling
    b = pickle.load(fp)'''

#Get RNALOCATE PPI Information

'''f = open('proteinIntDatabase.txt', 'r')
lines = [line.rstrip() for line in f]
f.close()

ppi = []
for item in lines:
    ppi.append(item.split())

myppi = []    
for item in ppi:
    for gene in gene_info:
        if(len(gene) >= 3): #We have protein information
            if(item[0] == gene[1]):
                temp.append(item[0])
                temp.append(item[1])
                temp.append(item[2])
                myppi.append(temp)
                temp = []
                break
  
f = open("myppiRNALOCATE.txt", "w")    #PPI of my Data
for item in myppi:
    f.write(str(item[0]))
    f.write("\t")
    f.write(str(item[1]))
    f.write("\t")
    f.write(str(item[2]))
    f.write("\n")
    
f.close()'''
 
#make protein protein reference
protRef = { gene_info[i][1] : i for i in range(0, len(gene_info)) }#make prot index for completing PPI        
organRef = { gene_info[i][0] : gene_info[i][2]  for i in range(0, len(gene_info)) } #map each gene to its organism

 
'''f = open("my_gene_info_unique.txt", "w")
for item in gene_info:
    for j in item:
        f.write(str(j))
        f.write("\t")
    f.write("\n")
f.close()'''
#Remove duplicate information for each RNA
'''gene_info = []
for i in gene_info_temp:
    item = list(set(i))
    gene_info.append(item)
  
print(gene_info)'''

#Remove NAN strings

'''f = open('ensemble_gene_ids.txt', 'r')
lines = [line.rstrip() for line in f]
f.close()

ppi = []
for item in lines:
    ppi.append(item.split())
    
print(len(ppi))

f = open('gene_ids.txt', 'r')
lines = [line.rstrip() for line in f]
f.close()

print(len(lines))'''