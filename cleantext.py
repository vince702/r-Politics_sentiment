#!/usr/bin/env python3


#VINCENT CHI
#304576879

"""Clean comment text for easier parsing."""

from __future__ import print_function

import re
import string
import argparse
import sys
import json


__author__ = ""
__email__ = ""

# Depending on your implementation,
# this data may or may not be useful.
# Many students last year found it redundant.
_CONTRACTIONS = {
    "tis": "'tis",
    "aint": "ain't",
    "amnt": "amn't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hell": "he'll",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "ol": "'ol",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "somebodys": "somebody's",
    "someones": "someone's",
    "somethings": "something's",
    "thatll": "that'll",
    "thats": "that's",
    "thatd": "that'd",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "wedve": "wed've",
    "well": "we'll",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatd": "what'd",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whodve": "whod've",
    "wholl": "who'll",
    "whore": "who're",
    "whos": "who's",
    "whove": "who've",
    "whyd": "why'd",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've"
}

# You may need to write regular expressions.

def sanitize(text):
    """Do parse the text in variable "text" according to the spec, and return
    a LIST containing FOUR strings 
    1. The parsed text.
    2. The unigrams
    3. The bigrams
    4. The trigrams
    """

    # YOUR CODE GOES BELOW:
    original = text

    #remove newlines and extra spaces
    text = re.sub(r'\n+|\t+', ' ', text)
    text = re.sub(r'\\n+|\\t+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

  	#remove urls and /r and /u links

    text = re.sub(r'[\(]?http[s]?\:\/\/\S+[\)]?|[\(]?www\.\S+[\)]?|][\(].*?[\)]', '', text)

    text = re.sub(r'\/(r\/\S+)', r'\1', text)

    text = re.sub(r'\/(u\/\S+)', r'\1', text)

    text = re.sub(r'[^a-zA-Z0-9\;\.\!\?\,\:\-\\\/\'\"\s]', '', text)


    #parse punc at end of word

    match = re.search(r'\S[^a-zA-Z0-9\s]\s|\S[^a-zA-Z0-9\s]$', text)
    while (match != None):
	    text = re.sub(r'([^a-zA-Z0-9\s])\s', r' \1 ', text)
	    
	    text = re.sub(r'([^a-zA-Z0-9\s])$', r' \1 ', text)
	    match = re.search(r'\S[^a-zA-Z0-9\s]\s|\S[^a-zA-Z0-9\s]$', text)

    #parse punc at front of word

    match = re.search(r'\s[^a-zA-Z0-9\s]\S|^[^a-zA-Z0-9\s]\S', text)
    while (match != None):
	    text = re.sub(r'(\s[^a-zA-Z0-9\s])', r' \1 ', text)
	    text = re.sub(r'(^[^a-zA-Z0-9\s])', r' \1 ', text)
	    match = re.search(r'\s[^a-zA-Z0-9\s]\S|^[^a-zA-Z0-9\s]\S', text)



    text = re.sub(r'([^a-zA-Z0-9\s]$)', r' \1', text)

    #remove duplicate spaces again
    text = re.sub(r'\s+', ' ', text)


    #list of separated words/ending punctuation
    text_list = text.split();

    #lower case everything
    text_list = [token.lower() for token in text_list]

    #=====CREATE PARSED TEXT======#
    text = ''
    #remove special characters 
    for i in text_list:
    	if len(i) == 1:
    		if re.search(r'[^a-zA-Z0-9\.\!\?\:\;\,]',i):
    			continue
    		else:
    			text += (i + ' ')
    	else:
    		text += i
    		text += ' '

    text = re.sub(r'\s$','',text)


    text_list = text.split();
    punctuation = string.punctuation


    #=======++CREATE N-GRAMS++=========#

    unigrams = ''
    
    for i in text_list:
    	if i not in punctuation:
    		unigrams += (i + ' ')

    unigrams = re.sub(r'\s$','',unigrams)



    bigrams = ''

    for i in range(len(text_list) - 1):
        if text_list[i] not in punctuation and text_list[i + 1] not in punctuation:
            bigrams += text_list[i] + '_' + text_list[i + 1]
            bigrams += ' '

    bigrams = re.sub(r'\s$','',bigrams)



    trigrams = ''

    for i in range(len(text_list) - 2):
        if text_list[i] not in punctuation and text_list[i + 1] not in punctuation and text_list[i + 2] not in punctuation:
            trigrams += text_list[i] + '_' + text_list[i + 1] + '_' + text_list[i + 2]
            trigrams += ' '

    trigrams = re.sub(r'\s$','',trigrams)
    #=======++END OF CREATE N-GRAMS++=========#

    parsed_text = text
   
   
    return [parsed_text, unigrams, bigrams, trigrams]


if __name__ == "__main__":
    # This is the Python main function.
    # You should be able to run
    # python cleantext.py <filename>
    # and this "main" function will open the file,
    # read it line by line, extract the proper value from the JSON,
    # pass to "sanitize" and print the result as a list.

    # YOUR CODE GOES BELOW.

    # We are "requiring" your write a main function so you can
    # debug your code. It will not be graded.

    print ('\n')
    text = sys.argv[1]

    #text = "That's a symptom you're highlighting!\n\n\"Countries where the portion of the voting population that are ignorant, uneducated or extremist, have no significant influence over their government or policy\" would be more accurate...\n\nUnfortunately in America, and most countries really, a huge portion of the voting population are aggressively stupid"
    l = sanitize(text)
    for i in l:
    	print(i)
    	print ('\n')
    exit(0)


    file_name = sys.argv[1]
    regex = re.compile(r'^.*[.](?P<ext>\w+)$')
    file_ext = regex.match(file_name).group('ext')

    if file_ext == 'json':
        with open(file_name, "r") as json_data:
            for line in json_data:
                data = json.loads(line)
                print(sanitize(data['body']))

    else:
        exit(1)
