#Edit Distance
def editDistDP(str1, str2, m, n):
	# Create a table to store results of subproblems
	dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

	# Fill d[][] in bottom up manner
	for i in range(m + 1):
		for j in range(n + 1):

			# If first string is empty, only option is to
			# insert all characters of second string
			if i == 0:
				dp[i][j] = j # Min. operations = j

			# If second string is empty, only option is to
			# remove all characters of second string
			elif j == 0:
				dp[i][j] = i # Min. operations = i

			# If last characters are same, ignore last char
			# and recur for remaining string
			elif str1[i-1] == str2[j-1]:
				dp[i][j] = dp[i-1][j-1]

			# If last character are different, consider all
			# possibilities and find minimum
			else:
				dp[i][j] = 1 + min(dp[i][j-1],	 # Insert
								dp[i-1][j],	 # Remove
								dp[i-1][j-1]) # Replace

	return dp[m][n]


# Driver code
str2 = "dof"
str1 = "dog"

print(editDistDP(str1, str2, len(str1), len(str2)))

import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize
text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
The sky is pinkish-blue. You shouldn't eat cardboard"""
tokenized_text=sent_tokenize(text)
print(tokenized_text)

#word token
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print(tokenized_word)

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stemmed_words=[]
for w in tokenized_word:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:",tokenized_word)
print("Stemmed Sentence:",stemmed_words)

nltk.download('wordnet')
nltk.download('omw-1.4')

#Lexicon Normalization
#performing stemming and Lemmatization

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

word = "flagged"
print("Lemmatized Word:",lem.lemmatize(word,"v"))
print("Stemmed Word:",stem.stem(word))


# Stemming is a process that stems or removes last few characters from a word, often leading to incorrect meanings and spelling. 
# Lemmatization considers the context and converts the word to its meaningful base form, which is called Lemma.
import nltk
from nltk.stem.porter import PorterStemmer
porter_stemmer  = PorterStemmer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
  print("Stemming for {} is {}".format(w,porter_stemmer.stem(w)))  
  
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
	print("Lemma for {} is {}".format(w, wordnet_lemmatizer.lemmatize(w)))  
 
# Import the toolkit and the full Porter Stemmer library
import nltk

from nltk.stem.porter import *
p_stemmer = PorterStemmer()
words = ['run','runner','running','ran','runs','easily','fairly']
for word in words:
    print(word+' --> '+p_stemmer.stem(word))

# Perform standard imports:
import spacy
nlp = spacy.load('en_core_web_sm')
def show_lemmas(text):
    for token in text:
        print(f'{token.text:{12}} {token.pos_:{6}} {token.lemma:<{22}} {token.lemma_}')
        
doc = nlp("I saw eighteen mice today!")
show_lemmas(doc)

import nltk
from nltk.stem import SnowballStemmer
SnowballStemmer.languages
('danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'norwegian', 'porter', 'portuguese', 'romanian', 'russian', 'spanish', 'swedish')
spanishstemmer=SnowballStemmer('spanish')
print(spanishstemmer.stem('comiendo'))

frenchstemmer=SnowballStemmer('french')
print(frenchstemmer.stem('manger'))