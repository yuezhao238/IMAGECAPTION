from sklearn.feature_extraction.text import TfidfVectorizer
import re
import json

data_json_path=f'../data/deepfashion-multimodal/train_captions.json'

documents = []

with open(data_json_path, 'r') as f:
    lines = json.load(f)

for line in lines.values():
    line = re.sub(r'[^\w\s]', ' ', line)
    line = line.lower()
    documents.append(line)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()
tfidf_weights = {}
for col in range(tfidf_matrix.shape[1]):
    tfidf_weights[feature_names[col]] = tfidf_matrix[0, col]

with open('tfidf_weights.json', 'w', encoding="utf8") as f:
    json.dump(tfidf_weights, f, indent=4, ensure_ascii=False)