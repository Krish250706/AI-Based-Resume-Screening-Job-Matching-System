import re 
import numpy as np 
import pandas as pd 
import spacy 
import joblib 
 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import TruncatedSVD 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import LabelEncoder 
from sklearn.neural_network import MLPClassifier 
 
print("\n[1] Loading dataset...") 
df = pd.read_csv("Cleaned_Resume.csv") 
df.columns = ['text', 'category'] 
df.dropna(inplace=True) 
 
df['candidate_id'] = df.index 
print("Dataset loaded:", len(df), "records") 
 
print("\n[2] Cleaning text...") 
def clean(text): 
    text = str(text).lower() 
    text = re.sub(r'[^a-z ]', ' ', text) 
    text = re.sub(r'\s+', ' ', text) 
    return text 
 
df['clean'] = df['text'].apply(clean) 
 
print("\n[3] Parsing resumes...") 
nlp = spacy.load("en_core_web_sm") 
 
SKILLS_DB = [ 
    "python", "java", "machine learning", "data science", 
    "sql", "deep learning", "nlp", "tensorflow", 
    "c++", "html", "css", 
    "pandas", "numpy", "keras", "statistics", "ai" 
] 
 

docs = list(nlp.pipe(df['clean'], batch_size=128)) 
 
skills_list = [] 
for doc in docs: 
    text = doc.text.lower() 
    skills = list(set([s for s in SKILLS_DB if s in text])) 
    skills_list.append(skills) 
 
df['skills'] = skills_list 
 
print("\n[4] Feature engineering...") 
df['skill_count'] = df['skills'].apply(len) 
df['text_length'] = df['clean'].apply(lambda x: len(x.split())) 
df['experience'] = df['clean'].str.count(r'\d+ years') 
 
print("\n[5] Feature extraction...") 
tfidf = TfidfVectorizer(max_features=2000) 
X_text = tfidf.fit_transform(df['clean']) 
 
svd = TruncatedSVD(n_components=50) 
X_text = svd.fit_transform(X_text) 
 
X = np.column_stack([ 
    X_text, 
    df['skill_count'], 
    df['text_length'], 
    df['experience'] 
]) 
 
print("\n[6] Clustering...") 
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10) 
df['cluster'] = kmeans.fit_predict(X) 
 
print("\n[7] Training ANN...") 
le = LabelEncoder() 
y = le.fit_transform(df['category']) 
 
ann = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300) 
ann.fit(X, y) 
 
print("\n[8] Saving processed data...")

df.to_csv("processed_resume.csv", index=False) 
np.save("features.npy", X) 
 
joblib.dump(tfidf, "tfidf.pkl") 
joblib.dump(svd, "svd.pkl") 
joblib.dump(ann, "ann.pkl") 
joblib.dump(le, "label_encoder.pkl") 
 
print("Preprocessing complete.") 
