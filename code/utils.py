#This script contains functions for data reading and feature generation. 
import argparse
import time
import math

import sys
import os
import numpy as np
import pickle

def basePair_Score(filepath, protein):
    basePairList = []
    tmpList = []
    count = 0
    with open(filepath, 'r') as file1:
        lines = file1.readlines()
        for line in lines:
            if line.startswith(">"):
                words = line.split("\t")
                rbpID = words[0].split("|")[2]
                if count != 4:
                    pairProba = words[1:-2]
                    tmpList.append([format(float(i), '.3f') for i in pairProba])
                else:
                    pairProba = words[2:-1]
                    pairProba.append(words[len(words) - 1].strip())
                    tmpList.append([format(float(i), '.3f') for i in pairProba])
                    basePairList.append(tmpList)
                count += 1
              
            else:
                tmpList = []
                count = 0

    tmp_arr = np.array(basePairList).transpose((0, 2, 1))
    return  tmp_arr

def read_fasta_file(protein):
    bindSiteDict_pos = dict()
    bindSiteDict_neg = dict()
    posLabel = []
    negLabel = []

    with open('PreviousData/positive.out', 'r') as posFile:
        lines = posFile.readlines()
        for line in lines:
            words = line.split("\t")
            bindID = words[0].strip()
            rbpID = words[0].split("|")[2]
            if bindID not in bindSiteDict_pos.keys():
                bindSiteDict_pos[bindID] = words[1] + "\t" + words[2] + "\t" + words[3].strip()
            posLabel.append((bindID, rbpID, 1))

    with open('PreviousData/negative.out', 'r') as posFile:
        lines = posFile.readlines()
        for line in lines:
            words = line.split("\t")
            bindID = words[0].strip()
            rbpID = words[0].split("|")[2]
            if bindID not in bindSiteDict_neg.keys():
                bindSiteDict_neg[bindID] = words[1] + "\t" + words[2] + "\t" + words[3].strip()
            negLabel.append((bindID, rbpID, 0))    

    return posLabel, negLabel, bindSiteDict_pos, bindSiteDict_neg

def getVocabIndex_pretrained(k_mer1):
   
    with open('PreviousData/vocab'+str(k_mer1)+'_w2c.txt', 'r') as fileVocab:
        lines = fileVocab.readlines()
        vocabList = [line.strip() for line in lines]
    vocabDict = dict((w.replace("T", "U"), i) for i, w in enumerate(vocabList))
    with open('PreviousData/Glove/gloveSeq_vector.pkl', 'rb') as file1:
        embedding_matrix = pickle.load(file1)
    return vocabDict, np.array(embedding_matrix)

def encodeSeqIndex(seqSet, k_mer, vocabIndex_dict):
    k = k_mer
    vecSet=[]
    for seq in seqSet:
        vec_index = []
        # seq = seq.replace("U", "T").upper()
        for x in range(len(seq) - k + 1):  # k determines the stride of slidding window
            kmer = seq[x:x + k]
            vec_index.append(vocabIndex_dict[kmer])
        vecSet.append(vec_index)

    return vecSet

def seqStructMapping(bindSiteList,posSeqStrucDict, negSeqStrucDict):
    seqList = []
    dotBrackList = []
    loopTypeList = []
    for i in bindSiteList:
        if i[0] in posSeqStrucDict.keys():
            seqList.append(posSeqStrucDict[i[0]].split("\t")[0].replace("T", "U").upper())
            dotBrackList.append(posSeqStrucDict[i[0]].split("\t")[1].strip())
            loopTypeList.append(posSeqStrucDict[i[0]].split("\t")[2].strip())
        elif i[0] in negSeqStrucDict.keys():
            seqList.append(negSeqStrucDict[i[0]].split("\t")[0].replace("T", "U").upper())
            dotBrackList.append(negSeqStrucDict[i[0]].split("\t")[1].strip())
            loopTypeList.append(negSeqStrucDict[i[0]].split("\t")[2].strip())
    return seqList, dotBrackList, loopTypeList

def getRBPBioChem():
    with open('data/rbpChemDict_stack.pkl', 'rb') as file5:
        rbpchem_dict = pickle.load(file5)

    return rbpchem_dict
