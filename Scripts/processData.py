
#GET FASTA seq and PPI ensemble

import Bio
from Bio import SeqIO
import pandas as pd
import numpy as np

def process_data():
    
    gene_ids_temp = []
    X = []
    y = []
    for seq_record in SeqIO.parse("Cytoplasm_train.fasta", "fasta"):
        gene_ids_temp.append(seq_record.id)
        X.append(seq_record.seq)
        y.append("Cytoplasm")
    for seq_record in SeqIO.parse("Endoplasmic_reticulum_train.fasta", "fasta"):
        gene_ids_temp.append(seq_record.id)
        X.append(seq_record.seq)
        y.append("Endoplasmic_reticulum")
        
    for seq_record in SeqIO.parse("Extracellular_region_train.fasta", "fasta"):
        gene_ids_temp.append(seq_record.id)
        X.append(seq_record.seq)
        y.append("Extracellular_region")
        
    for seq_record in SeqIO.parse("Mitochondria_train.fasta", "fasta"):
        gene_ids_temp.append(seq_record.id)
        X.append(seq_record.seq)
        y.append("Mitochondria")
        
    for seq_record in SeqIO.parse("Nucleus_train.fasta", "fasta"):
        gene_ids_temp.append(seq_record.id)
        X.append(seq_record.seq) 
        y.append("Nucleus")
     
    gene_ids = []
    mrnaloc = [] 
    for ids in gene_ids_temp:
        gene_ids.append((str(ids).split("#")[1]))
        mrnaloc.append((str(ids).split("#")[0]))
                        
    
    print("Number of gene ids", len(gene_ids))
    new = list(set(gene_ids))
    print("Unique gene ids ", len(new))
    print("Number of all mRNA's", len(X))   
    print("length of y", len(y)) 
    
    data = pd.read_excel('organismUnique.xlsx')
    df = pd.DataFrame(data, columns= ['Entry','Status','Entry name','Cross-reference (GeneID)','Cross-reference (STRING)','Organism','Interacts with'])
    arr = df.to_numpy()
    
    #Extract gene Information from genes
    f = open('gene_ids.txt', 'r')
    genes = [line.rstrip() for line in f]
    f.close()
    
    gene_info = [] #Contains Entrez gene id, string Prot id, Organism
    temp = []
    for gen in genes:
        temp.append(gen)
        for item in arr:
            #print(item[3].split(";")[0])
            if( (item[3].split(";")[0]) == gen ): #gene id
                temp.append( str(item[4]).split(";")[0] ) #prot id
                temp.append(item[5]) #Organism
                break
                
        
        gene_info.append(temp)
        temp = []
    
    #build protein reference for PPI extraction    
    f = open('gene_ids_unique.txt', 'r')
    genes = [line.rstrip() for line in f]
    f.close()
    
    prot_info = [] #Contains Entrez gene id, string Prot id, Organism
    temp = []
    for gen in genes:
        temp.append(gen)
        for item in arr:
            #print(item[3].split(";")[0])
            if( (item[3].split(";")[0]) == gen ): #gene id
                temp.append( str(item[4]).split(";")[0] ) #prot id
                temp.append(item[5]) #Organism
                break
                
        
        prot_info.append(temp)
        temp = []
     
    remove_indices = []
    for ind, item in enumerate(prot_info):
        if(len(item) != 3 or item == None):
            remove_indices.append(ind)
            
    #prot_info_new = np.asanyarray(prot_info)
    prot_info_new = [i for j, i in enumerate(prot_info) if j not in remove_indices] 
    prot_info = prot_info_new
    
    prot_info = np.asarray(prot_info)
    print(prot_info)
    protRef = { prot_info[i][1] : i for i in range(0, len(prot_info)) }#make prot index for completing PPI 
    
    with open('myppiRNALOCATE.txt', 'r') as f:
        locations = [line.rstrip("\n") for line in f]
    myppi = []
    for item in locations:
        myppi.append(item.split("\t"))
        
        
    ppiMatrix = np.zeros( (len(prot_info),len(prot_info)))
    for prot in prot_info:
            for protInt in myppi:
                if(protInt[0] == prot[1]):
                    if str((protInt[1])) in protRef:
                        ppiMatrix[protRef[protInt[0]]][protRef[protInt[1]]] = protInt[2]
                        ppiMatrix[protRef[protInt[1]]][protRef[protInt[0]]] = protInt[2]
                        
    rnalocPPI = np.zeros( (len(gene_ids),len(gene_ids)) )
    
    only_prot_info = []
    for item in gene_info:
        if(len(item) == 3): #we have protein information
            only_prot_info.append(item[1])
        else:
            only_prot_info.append(item[0])
            
    for ppi_row, key in enumerate( protRef):
        res_list = [i for i, value in enumerate(only_prot_info) if value == key]
        for j in res_list:
            for ppi_col in range(len(protRef)):
                rnalocPPI[j][ppi_col] = ppiMatrix[ppi_row][ppi_col]
            
    print("shape of gene info", np.shape(gene_info))
    print("shape of X DATA", len(X))
    print("shape of PPI", np.shape(rnalocPPI))
    
    return X, y, rnalocPPI, gene_info

