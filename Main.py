import heapq 
import heapq 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from collections import deque 
import joblib 
import re 
 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics.pairwise import cosine_similarity 
 
print("\n[1] Loading processed data...") 
df = pd.read_csv("processed_resume.csv") 
X = np.load("features.npy") 
 
tfidf = joblib.load("tfidf.pkl") 
svd = joblib.load("svd.pkl") 
ann = joblib.load("ann.pkl") 
le = joblib.load("label_encoder.pkl") 
 
print("Data loaded:", len(df), "records") 
 
SKILLS_DB = [ 
    "python", "java", "machine learning", "data science", 
    "sql", "deep learning", "nlp", "tensorflow", 
    "c++", "html", "css", 
    "pandas", "numpy", "keras", "statistics", "ai" 
]

print("\n[2] Job matching...") 
job_desc = input("Enter Job Description: ") 
 
job_clean = str(job_desc).lower() 
job_clean = re.sub(r'[^a-z ]', ' ', job_clean) 
job_clean = re.sub(r'\s+', ' ', job_clean) 
 
job_skills = [s for s in SKILLS_DB if s in job_clean] 
 
job_vec = tfidf.transform([job_clean]) 
job_vec = svd.transform(job_vec) 
 
job_extra = np.array([[len(job_skills), len(job_clean.split()), 0]]) 
job_vec = np.hstack([job_vec, job_extra]) 
 
job_sim = cosine_similarity(job_vec, X).flatten() * 100 
 
ann_probs = ann.predict_proba(X) 
ann_scores = ann_probs.max(axis=1) * 100 
 
match_counts = [] 
for i in range(len(df)): 
    skills = eval(df.loc[i, 'skills']) 
    count = len(set(skills) & set(job_skills)) 
    match_counts.append(count) 
 
df['match_count'] = match_counts 
 
reg = LinearRegression() 
reg.fit(X, job_sim) 
reg_scores = reg.predict(X) 
reg_scores = ((reg_scores - reg_scores.min()) / (reg_scores.max() - 
reg_scores.min() + 1e-9)) * 100 
 
df['score'] = ( 
    0.3 * job_sim + 
    0.2 * reg_scores + 
    0.4 * (np.array(match_counts) * 10) + 
    0.1 * ann_scores 
)

mask = df['match_count'] >= 3 
 
if mask.sum() == 0: 
    mask = df['match_count'] >= 1 
 
df = df[mask].reset_index(drop=True) 
X = X[mask] 
 
print("\n[3] Graph...") 
sim = cosine_similarity(X) 
 
graph = {} 
for i in range(len(sim)): 
    graph[i] = sim[i].argsort()[-6:-1].tolist() 
 
print("\n[4] BFS...") 
start = np.argmax(df['score']) 
 
def bfs(start): 
    visited = set() 
    queue = deque([start]) 
    result = [] 
 
    while queue: 
        node = queue.popleft() 
        if node not in visited: 
            visited.add(node) 
            result.append(node) 
            queue.extend(graph[node]) 
    return result 
 
print("BFS:", bfs(start)[:5]) 
 
print("\n[5] A*...") 
def heuristic(node): 
    return 1 - (df.loc[node, 'score'] / 100) 
 
def astar(start): 
    pq = [(0, start)] 
    visited = set() 
    best = start 

while pq: 
        cost, node = heapq.heappop(pq) 
 
        if node in visited: 
            continue 
 
        visited.add(node) 
 
        if df.loc[node, 'score'] > df.loc[best, 'score']: 
            best = node 
 
        for n in graph[node]: 
            heapq.heappush(pq, (cost + heuristic(n), n)) 
 
    return best 
best = astar(start) 
 
print("\n[6] Results...") 
df = df.sort_values(by='score', ascending=False) 
 
print(df[['candidate_id', 'category', 'score']].head(5)) 
 
top = df.iloc[0] 
matched = list(set(eval(top['skills'])) & set(job_skills)) 
 
print("\nBest Candidate:") 
print("ID:", top['candidate_id']) 
print("Category:", top['category']) 
print("Score:", round(top['score'], 2)) 
print("Matched Skills:", matched) 
 
print("\n[7] Graphs...") 
 
plt.figure() 
plt.hist(df['score'], bins=10) 
plt.title("Score Distribution") 
plt.show() 
 
plt.figure() 
df['cluster'].value_counts().plot(kind='bar') 
plt.title("Cluster Distribution") 
plt.show() 

