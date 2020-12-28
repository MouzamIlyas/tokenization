from nltk.stem import PorterStemmer
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."

print(sent_tokenize(EXAMPLE_TEXT))
print(word_tokenize(EXAMPLE_TEXT))
words = word_tokenize(EXAMPLE_TEXT)
ps = PorterStemmer()
# print(EXAMPLE_TEXT)
tagged = nltk.pos_tag("EXAMPLE_TEXT")
print(tagged[:20])
for w in words:
	rootWord=ps.stem(w)
	print(rootWord)