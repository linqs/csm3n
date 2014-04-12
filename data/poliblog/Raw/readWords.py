#!/usr/bin/python

import os
import re

filePat = re.compile("([0-9]+).txt")

wordIndex = 1

words = dict()

for crawl in ['02092012', '05092102']:
	filenames = dict()

	for i in os.listdir('%s/' % crawl):
		match = filePat.match(i)

		if match:
			filenames[int(match.group(1))] = i


	wordIJ = open("wordIJ_%s.txt" % crawl, 'w')

	for i in filenames:
		for line in open("%s/%s"% (crawl, filenames[i]), 'r'):
			fileWords = line.strip().split(' ')
			for word in fileWords:
				if word not in words:
					words[word] = wordIndex
					wordIndex += 1

				wordIJ.write("%d %d\n" % (i, words[word]))

	wordIJ.close()


dictionary = open("wordIndex.txt", 'w')

for word in words:
	dictionary.write("%d %s\n" %(words[word], word))

dictionary.close()
