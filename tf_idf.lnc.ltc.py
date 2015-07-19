# -*- coding: utf-8 -*-

import os
import operator
import numpy as np


termFreqDict = {} # term frequency: (term, tf)
docFreqDict = {} # doc frequency: (term, df)
invDocFreqDict = {} # inverse document frequency: (term, idf)

docNum = 0 # number of documents
topTermNum = 2000 # top frequency term to keep
topTermName2Order = {} # term name and its order

docId2CateDict = {} # doc id to category
docId2ContentDict = {} # doc id to content
indexDict = {} # doc id to vetorize

def readFile(name):
	idx = 0
	with open(name, 'r') as fr:
		for line in fr:
			line = line.strip('\n').strip('\r')
			curList = line.split('\t', 1)
			docId2CateDict[idx] = curList[0]
			docId2ContentDict[idx] = curList[1]
			idx += 1
	# print 'idx:', idx
	return idx
	
	
# get term frequency		
def getTermFreqency():
	for id in docId2ContentDict:
		content = docId2ContentDict[id]
		termList = content.split(' ')
		for term in termList:
			termFreqDict[term] = termFreqDict.get(term, 0) + 1


# get top term for vectorize document			
def getTopTerm():
	sortedTermList = sorted(termFreqDict.items(), key = operator.itemgetter(1), reverse = True)
	for i in xrange(topTermNum):
		topTermName2Order[ sortedTermList[i][0] ] = i


# get document and inverse frequency			
def getDocAndInvFreqency():
	# get document frequency
	for id in docId2ContentDict:
		content = docId2ContentDict[id]
		termSet = set( content.split(' ') )
		for term in termSet:
			docFreqDict[term] = docFreqDict.get(term, 0) + 1
	# get inverse document frequency
	for term in docFreqDict:
		invDocFreqDict[term] = np.log10(docNum*1.0 / docFreqDict[term])

		
# vectorize index file
# return tuple: position, tf-idf	
def vectorizeIndexFile(content):
	retList = list()
	termList = content.split(' ')
	
	# get term-doc frequency: tf_{t,d}
	termFreqDocDict = {}
	for term in termList:
		if term in topTermName2Order:
			termFreqDocDict[term] = termFreqDocDict.get(term, 0) + 1
	
	# normalize tf*idf
	cnt = 0.0
	for term in termFreqDocDict:
		# tf_idf = termFreqDocDict[term] * invDocFreqDict[term]
		tf = 1 + np.log10( termFreqDocDict[term] * 1.0 )
		cnt += (tf * tf)
	cnt = np.sqrt(cnt)
	
	for term in termFreqDocDict:
		# tf_idf = termFreqDocDict[term] * invDocFreqDict[term]
		tf = 1 + np.log10( termFreqDocDict[term] * 1.0 )
		retList.append( (topTermName2Order[term], tf / cnt) )
	retList.sort(key = lambda e: e[0] )
	return retList
	
	
# vectorize index file
def vectorizeQuery(query):
	queryVecList = list()
	freqDict = {}
	termList = query.split(' ')
	#print 'queryy term number:', len(termList)
	for term in termList:
		if term in topTermName2Order:
			freqDict[term] = freqDict.get(term, 0) + 1
	# normalize
	cnt = 0.0
	for term in freqDict:
		tf_idf = freqDict[term] * invDocFreqDict[term]
		cnt += ( tf_idf * tf_idf )
	cnt = np.sqrt(cnt)
		
	for term in freqDict:
		tf_idf = freqDict[term] * invDocFreqDict[term]
		queryVecList.append( (topTermName2Order[term], tf_idf * tf_idf / cnt) )
	queryVecList.sort(key = lambda e: e[0])
	return queryVecList

	
def createIndex():
	for id in docId2ContentDict:
		indexDict[id] = vectorizeIndexFile( docId2ContentDict[id] )


# calculate cosine distance of two list.
# the lists have been normlized.		
def cosineDistance(aList, bList):
	a, b = len(aList), len(bList)
	i, j = 0, 0
	cnt = 0.0
	while i < a and j < b:
		if aList[i][0] == bList[j][0]:
			cnt += ( aList[i][1] * bList[j][1] )
			i += 1
			j += 1
		elif aList[i][0] < bList[j][0]:
			i += 1
		else:
			j += 1
	return cnt


# seearch relevant document	
def searchDoc(query):
	queryVec = vectorizeQuery(query)
	scoreDict = {} # doc id to relevance score
	for id in indexDict:
		scoreDict[id] = cosineDistance(queryVec, indexDict[id])
	scoreList = sorted(scoreDict.items(), key = operator.itemgetter(1), reverse = True)
	
	# get category of doc
	resultList = list()
	for e in scoreList:
		resultList.append( docId2CateDict[e[0]] )
	
	# fw = open('score-result.txt', 'w')
	# for e in scoreList:
		# id = e[0] # document id
		# score = e[1] # relevance score
		# cate = docId2CateDict[id] # document category
		# fw.write( 'cate:' + cate + '\t id:' + str(id) + '\t score:' + str(score) + '\n' )
	# fw.flush()
	# fw.close()
	return resultList


# calculate average precision	
def calAveragePrecision(category, resultList):
	n = len(resultList)
	hit, cnt = 0.0, 0.0
	for i in xrange(n):
		if resultList[i] == category:
			hit += 1.0
			cnt += ( hit/(i + 1) )
	cnt /= hit
	return cnt


# write dict to debug
def writeDict(aDict, pathName):
	with open(pathName, 'w') as fw:
		for key in aDict:
			fw.write( str(key) + '\t' + str(aDict[key]) + '\n' )
		fw.flush()
	
if __name__ == '__main__':
	trainFileName = '20ng/20ng-train-stemmed.txt'
	# load in file
	docNum = readFile(trainFileName)
	print 'document number:', docNum
	# get tf*idf
	getTermFreqency()
	getTopTerm()
	getDocAndInvFreqency()
	writeDict(termFreqDict, '20ng/termFreqDict.txt')
	writeDict(topTermName2Order, '20ng/topTermName2Order.txt')
	writeDict(docFreqDict, '20ng/docFreqDict.txt')
	writeDict(invDocFreqDict, '20ng/invDocFreqDict.txt')
	
	# create index
	createIndex()
	# evaluation
	queryNum, meanAvgPre = 0, 0.0
	queryFileName = '20ng/20ng-test-stemmed.txt'
	with open(queryFileName, 'r') as fr:
		for line in fr:
			curList = line.strip('\n').strip('\r').split('\t')
			resultList = searchDoc(curList[1])
			avgPre = calAveragePrecision(curList[0], resultList)
			print 'average precision:', avgPre, '\t category:', curList[0]
			meanAvgPre += avgPre
			queryNum += 1
	meanAvgPre /= queryNum
	print 'mean average precision:', meanAvgPre
