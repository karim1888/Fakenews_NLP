import pandas as pd  # For loading, cleaning, and manipulating datasets (DataFrames)
import numpy as np   # For numerical operations and array handling
import html          # To decode HTML entities in text
import re            # For regular expressions (text pattern cleaning)
import string        # For string operations like punctuation lists
from collections import Counter   # For counting word or tag frequencies
import matplotlib.pyplot as plt   # For creating basic charts and plots
import seaborn as sns             # For advanced and beautiful data visualizations

# NLP Libraries
import nltk                                  # Natural Language Toolkit (main NLP library)
from nltk.corpus import stopwords            # Common stop words (e.g., 'the', 'is')
from nltk.tokenize import word_tokenize, sent_tokenize  # Tokenization into words/sentences
from nltk.stem import WordNetLemmatizer      # For lemmatizing words (e.g., "running" → "run")
from nltk import pos_tag, ne_chunk           # Part-of-speech tagging & Named Entity Recognition
from nltk.probability import FreqDist        # For word frequency distribution
from nltk.sentiment import SentimentIntensityAnalyzer  # Sentiment scoring (positive/negative/neutral)
from nltk.tree import Tree                   # To represent named entities in tree form

# Machine Learning Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# TfidfVectorizer: Convert text into TF-IDF features (term frequency-inverse document frequency)
# CountVectorizer: Convert text into raw count of words (Bag-of-Words model)

from sklearn.decomposition import LatentDirichletAllocation, NMF
# For topic modeling: LDA (Latent Dirichlet Allocation) and NMF (Non-negative Matrix Factorization)

from sklearn.model_selection import train_test_split
# For splitting data into training and testing sets

from sklearn.ensemble import RandomForestClassifier
# Random Forest model for text classification

from sklearn.metrics import classification_report, accuracy_score
# To evaluate model performance (precision, recall, F1-score, accuracy)

from sklearn.pipeline import Pipeline
# To chain multiple steps (cleaning, vectorizing, modeling) into one reusable pipeline

from sklearn.base import BaseEstimator, TransformerMixin
# To create custom transformers (like your `Tekonizer` class for text preprocessing)


# Import additional required libraries
from transformers import BertTokenizer, BertModel, pipeline  
# - BertTokenizer: Tokenizes text for BERT models (breaks text into words/subwords)
# - BertModel: Provides BERT's deep learning architecture for text embeddings
# - pipeline: Easy-to-use interface for pre-trained NLP models (e.g., sentiment analysis)

import torch  # PyTorch library for deep learning operations (required for BERT)

# Text Readability Metrics
from textstat import flesch_reading_ease, smog_index, dale_chall_readability_score  
# - flesch_reading_ease: Measures how easy text is to read (lower = harder)
# - smog_index: Estimates years of education needed to understand text  
# - dale_chall_readability_score: Uses familiar word lists to assess readability

# Topic Modeling
from bertopic import BERTopic  
# - Advanced topic modeling that uses BERT embeddings to discover topics in text

# Network Analysis
import networkx as nx  
# - Creates and analyzes network/graph structures (used for entity co-occurrence networks)
from itertools import combinations  
# - Generates all possible pairs of items (used for building co-occurrence networks)

# Dimensionality Reduction
from sklearn.decomposition import PCA  
# - Principal Component Analysis: Reduces high-dimension data (like BERT embeddings) to 2D/3D for visualization

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

# Tekonization class for text processing
class Tekonizer(BaseEstimator, TransformerMixin):
    def __init__(self, remove_stopwords=True, lemmatize=True, min_word_length=3):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        self.stop_words = set(stopwords.words('english'))
        # Keep important negation words for sentiment analysis
        self.stop_words.difference_update(['no', 'not', 'nor', 'neither'])
        self.lemmatizer = WordNetLemmatizer()
        
    def tek_tokenize(self, text):
        # HTML unescape
        text = html.unescape(text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Filter tokens
        tokens = [t for t in tokens if len(t) >= self.min_word_length]
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens
        
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [' '.join(self.tek_tokenize(text)) for text in X]

# Load datasets
true_df = pd.read_csv("C:/Users/PCD/Desktop/fake news/True.csv")
fake_df = pd.read_csv("C:/Users/PCD/Desktop/fake news/Fake.csv")

# Add labels
true_df['label'] = 'real'
fake_df['label'] = 'fake'

# Combine into one dataset
full_df = pd.concat([true_df, fake_df], ignore_index=True)

# Create balanced sample
real_sample = true_df.sample(n=2500, random_state=42)
fake_sample = fake_df.sample(n=2500, random_state=42)
df = pd.concat([real_sample, fake_sample], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Initialize Tekonizer and create clean_text column
tekonizer = Tekonizer()
df['clean_text'] = tekonizer.transform(df['text'])

# Frequent Analysis
def frequent_analysis(texts, n=20):
    all_tokens = [word for text in texts for word in word_tokenize(text)]
    freq_dist = FreqDist(all_tokens)
    return freq_dist.most_common(n)

# Get most frequent words for real and fake news
real_freq = frequent_analysis(df[df['label'] == 'real']['clean_text'])
fake_freq = frequent_analysis(df[df['label'] == 'fake']['clean_text'])

print("Most frequent words in real news:")
print(real_freq)
print("\nMost frequent words in fake news:")
print(fake_freq)

# Term Weighting with TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])

def get_top_terms(vectorizer, matrix, n=10):
    sums = matrix.sum(axis=0)
    data = []
    for col, term in enumerate(vectorizer.get_feature_names_out()):
        data.append((term, sums[0,col]))
    ranked = sorted(data, key=lambda x: x[1], reverse=True)
    return ranked[:n]

top_terms = get_top_terms(tfidf_vectorizer, tfidf_matrix)
print("\nTop TF-IDF weighted terms:")
print(top_terms)

# End-to-EndR (End-to-End Relationship extraction)
def extract_relationships(text):
    sentences = sent_tokenize(text)
    relationships = []
    for sent in sentences:
        tokens = word_tokenize(sent)
        tagged = pos_tag(tokens)
        entities = ne_chunk(tagged)
        
        # Extract named entities
        nes = []
        for chunk in entities:
            if isinstance(chunk, Tree):
                nes.append((' '.join([token for token, pos in chunk.leaves()]), chunk.label()))
        
        # Simple relationship extraction (subject-verb-object)
        if len(nes) >= 2:
            relationships.append((nes[0][0], "related_to", nes[1][0]))
    return relationships

# Apply to a sample
sample_text = df.iloc[0]['text']
print("\nRelationship extraction example:")
print(extract_relationships(sample_text))

# POS Tagging Analysis
def pos_analysis(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    pos_counts = Counter(tag for word, tag in tagged)
    return pos_counts

df['pos_tags'] = df['clean_text'].apply(pos_analysis)

# Print POS Tagging results for first 5 articles
print("\nPOS Tagging Analysis (First 5 Articles):")
for i, pos_tags in enumerate(df['pos_tags'].head()):
    print(f"\nArticle {i+1} POS Tags:")
    for tag, count in pos_tags.most_common():
        print(f"{tag}: {count}")

# Information Extraction (Named Entity Recognition)
def extract_entities(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    entities = ne_chunk(tagged)
    return [(' '.join([token for token, pos in chunk.leaves()]), chunk.label()) 
            for chunk in entities if isinstance(chunk, Tree)]

df['entities'] = df['text'].apply(extract_entities)

# Print Named Entities for first 5 articles
print("\n\nNamed Entity Recognition (First 5 Articles):")
for i, entities in enumerate(df['entities'].head()):
    print(f"\nArticle {i+1} Entities:")
    for entity, label in entities:
        print(f"{entity} ({label})")

# Author Profiling (simple version based on writing style)
def author_profiling_features(text):
    features = {}
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Basic stats
    features['num_sentences'] = len(sentences)
    features['num_words'] = len(words)
    features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
    
    # POS ratios
    pos_tags = [tag for word, tag in pos_tag(words)]
    pos_counts = Counter(pos_tags)
    total_pos = sum(pos_counts.values())
    
    for tag, count in pos_counts.items():
        features[f'pos_ratio_{tag}'] = count / total_pos if total_pos > 0 else 0
    
    # Punctuation analysis
    features['num_exclamation'] = text.count('!')
    features['num_question'] = text.count('?')
    
    return features

# Apply author profiling
author_features = pd.DataFrame([author_profiling_features(text) for text in df['text']])
df = pd.concat([df, author_features], axis=1)

# Print Author Profiling features for first 5 articles
print("\n\nAuthor Profiling Features (First 5 Articles):")
print(author_features.head())

# Topic Classification with LDA
def perform_topic_modeling(texts, n_topics=5):
    # Vectorize with count vectorizer
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=5000)
    count_matrix = count_vectorizer.fit_transform(texts)
    
    # LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(count_matrix)
    
    # Get top words for each topic
    feature_names = count_vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topics.append((f"Topic {topic_idx}", top_features))
    
    # Assign dominant topic to each document
    topic_distributions = lda.transform(count_matrix)
    dominant_topics = topic_distributions.argmax(axis=1)
    
    return topics, dominant_topics

n_topics = 3
topics, dominant_topics = perform_topic_modeling(df['clean_text'], n_topics)
df['dominant_topic'] = dominant_topics

# Print Topic Modeling results
print("\n\nTopic Modeling Results:")
print("\nDiscovered Topics:")
for topic in topics:
    print(f"\n{topic[0]}: {', '.join(topic[1])}")

print("\nTopic Distribution in Dataset:")
print(df['dominant_topic'].value_counts())

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    return sia.polarity_scores(text)

df['sentiment'] = df['text'].apply(analyze_sentiment)
df['sentiment_compound'] = df['sentiment'].apply(lambda x: x['compound'])
df['sentiment_label'] = df['sentiment_compound'].apply(
    lambda x: 'positive' if x >= 0.05 else 'negative' if x <= -0.05 else 'neutral')

# Print Sentiment Analysis results
print("\n\nSentiment Analysis Results:")
print("\nFirst 5 Articles Sentiment Scores:")
print(df[['text', 'sentiment_compound', 'sentiment_label']].head())

print("\nOverall Sentiment Distribution:")
print(df['sentiment_label'].value_counts())

# Print final dataframe structure
print("\n\nFinal DataFrame Structure:")
print(df.info())

# Print first 5 rows of final dataframe
print("\nFirst 5 Rows of Final DataFrame:")
print(df.head())


# 1. Word Frequency
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
real_words = [word for word, count in real_freq]
real_counts = [count for word, count in real_freq]
fake_words = [word for word, count in fake_freq]
fake_counts = [count for word, count in fake_freq]
plt.show()

x = np.arange(len(real_words))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, real_counts, width, label='Real News')
plt.bar(x + width/2, fake_counts, width, label='Fake News')
plt.xticks(x, real_words, rotation=45)
plt.title('Top Word Frequency Comparison')
plt.legend()
plt.show()

# 2. Sentiment Distribution
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 2)
sns.boxplot(x='label', y='sentiment_compound', data=df)
plt.title('Sentiment Distribution by News Type')
plt.show()

# 3. Topic Distribution
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 3)
topic_counts = df.groupby(['dominant_topic', 'label']).size().unstack()
topic_counts.plot(kind='bar', stacked=True)
plt.title('Topic Distribution by News Type')
plt.xticks(rotation=0)
plt.show()


# 4. Author Profiling Features
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 4)
sns.scatterplot(x='avg_sentence_length', y='pos_ratio_NN', hue='label', data=df)
plt.title('Author Profiling: Sentence Length vs Noun Ratio')

plt.tight_layout()
plt.show()

# Machine Learning Pipeline for Classification
X = df['text']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('preprocess', Tekonizer()),
    ('vectorize', TfidfVectorizer(max_features=5000)),
    ('classify', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Feature Importance
feature_importances = pipeline.named_steps['classify'].feature_importances_
feature_names = pipeline.named_steps['vectorize'].get_feature_names_out()
top_features = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)[:20]

print("\nTop Important Features for Classification:")
for feature, importance in top_features:
    print(f"{feature}: {importance:.4f}")
    
    

# Set style for visualizations
sns.set()
sns.set_palette("husl")

# ======================
# 1. Advanced Visualizations
# ======================

print("\nGenerating advanced visualizations...")

# 1. Sentiment Analysis by News Type (Violin Plot)
plt.figure(figsize=(10, 6))
plt.subplot(3, 2, 1)
sns.violinplot(x='label', y='sentiment_compound', data=df, inner="quartile", palette="Set2")
plt.title('Sentiment Distribution Density by News Type', fontsize=14)
plt.xlabel('News Type', fontsize=12)
plt.ylabel('Sentiment Compound Score', fontsize=12)
plt.show()

# 2. Topic Modeling Visualization (Heatmap)
plt.figure(figsize=(10, 6))
plt.subplot(3, 2, 2)
topic_matrix = pd.crosstab(df['dominant_topic'], df['label'], normalize='index')
sns.heatmap(topic_matrix, annot=True, fmt=".1%", cmap="YlGnBu")
plt.title('Topic Distribution by News Type', fontsize=14)
plt.xlabel('News Type', fontsize=12)
plt.ylabel('Dominant Topic', fontsize=12)
plt.show()


# 3. Readability Comparison
def calculate_readability(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    chars_per_word = np.mean([len(word) for word in words])
    words_per_sentence = np.mean([len(word_tokenize(sent)) for sent in sentences])
    return chars_per_word * words_per_sentence

df['readability'] = df['text'].apply(calculate_readability)

plt.figure(figsize=(10, 6))
plt.subplot(3, 2, 3)
sns.boxplot(x='label', y='readability', data=df, palette="Set3")
plt.title('Readability Comparison (Chars/Word × Words/Sentence)', fontsize=14)
plt.xlabel('News Type', fontsize=12)
plt.ylabel('Readability Score', fontsize=12)
plt.show()


# 4. Named Entity Frequency Comparison
plt.figure(figsize=(10, 6))
plt.subplot(3, 2, 4)
entity_counts_real = sum(df[df['label']=='real']['entities'].apply(lambda x: Counter([label for (text, label) in x])), Counter())
entity_counts_fake = sum(df[df['label']=='fake']['entities'].apply(lambda x: Counter([label for (text, label) in x])), Counter())
pd.DataFrame({'Real': entity_counts_real, 'Fake': entity_counts_fake}).plot(kind='bar', ax=plt.gca())
plt.title('Named Entity Frequency by News Type', fontsize=14)
plt.xlabel('Entity Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.show()


# 5. Emotion Analysis (Radar Chart)
emotion_df = df['text'].apply(lambda x: pd.Series(sia.polarity_scores(x)))
emotion_df.columns = ['Negative', 'Neutral', 'Positive', 'Compound']
emotion_df['label'] = df['label']
emotion_means = emotion_df.groupby('label').mean()

categories = list(emotion_means.columns)
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(10, 6))
plt.subplot(3, 2, 5, polar=True)
for label in emotion_means.index:
    values = emotion_means.loc[label].values.flatten().tolist()
    values += values[:1]
    plt.plot(angles, values, linewidth=1, linestyle='solid', label=label)
    plt.fill(angles, values, alpha=0.1)

plt.title('Emotion Radar Chart by News Type', fontsize=14, y=1.1)
plt.xticks(angles[:-1], categories)
plt.yticks(color="grey", size=7)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()

# 6. Network Graph of Co-occurring Entities
def build_cooccurrence_network(entities_list):
    G = nx.Graph()
    for entities in entities_list:
        entities = [e[0] for e in entities]
        for a, b in combinations(set(entities), 2):
            if G.has_edge(a, b):
                G[a][b]['weight'] += 1
            else:
                G.add_edge(a, b, weight=1)
    return G

fake_entities = df[df['label']=='fake']['entities'].tolist()
G_fake = build_cooccurrence_network(fake_entities)
top_nodes = sorted(G_fake.degree(weight='weight'), key=lambda x: x[1], reverse=True)[:15]
G_fake_filtered = G_fake.subgraph([n[0] for n in top_nodes])

plt.figure(figsize=(10, 6))
plt.subplot(3, 2, 6)
pos = nx.spring_layout(G_fake_filtered, k=0.5)
nx.draw_networkx_nodes(G_fake_filtered, pos, 
                      node_size=[v*10 for v in dict(G_fake_filtered.degree(weight='weight')).values()], 
                      node_color='lightcoral', alpha=0.8)
nx.draw_networkx_edges(G_fake_filtered, pos, edge_color='gray', alpha=0.5)
nx.draw_networkx_labels(G_fake_filtered, pos, font_size=10)
plt.title('Entity Co-occurrence Network in Fake News', fontsize=14)
plt.axis('off')

plt.tight_layout()
plt.show()

# ======================
# 2. Advanced NLP Techniques
# ======================

print("\nRunning advanced NLP techniques...")

# 1. BERT Embeddings Visualization (on sample)
print("Generating BERT embeddings visualization (this may take a few minutes)...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Sample the full DataFrame instead of just text column
sample_df = df.sample(100, random_state=42)
sample_texts = sample_df['text'].tolist()
sample_labels = sample_df['label'].tolist()

bert_embeddings = np.array([get_bert_embeddings(text) for text in sample_texts])

pca = PCA(n_components=2)
bert_2d = pca.fit_transform(bert_embeddings)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=bert_2d[:, 0], y=bert_2d[:, 1], 
                hue=sample_labels, palette="Set1")
plt.title('BERT Embeddings Visualization (PCA Reduced)', fontsize=14)
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.legend(title='News Type')
plt.show()

# 2. Stylometric Analysis
print("Performing stylometric analysis...")
def stylometric_features(text):
    words = word_tokenize(text)
    if len(words) == 0:
        return pd.Series({
            'flesch': 0,
            'smog': 0,
            'dale_chall': 0,
            'avg_word_length': 0,
            'type_token_ratio': 0
        })
    
    features = {
        'flesch': flesch_reading_ease(text),
        'smog': smog_index(text),
        'dale_chall': dale_chall_readability_score(text),
        'avg_word_length': np.mean([len(word) for word in words]),
        'type_token_ratio': len(set(words)) / len(words)
    }
    return pd.Series(features)


stylo_df = df['text'].apply(stylometric_features)
stylo_df['label'] = df['label']

plt.figure(figsize=(10, 6))
for i, col in enumerate(stylo_df.columns[:-1]):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='label', y=col, data=stylo_df)
    plt.title(f'Distribution of {col}', fontsize=12)
    plt.xlabel('')
plt.suptitle('Stylometric Feature Comparison by News Type', fontsize=16)
plt.tight_layout()
plt.show()

# 3. BERTopic Modeling (on sample)
print("Running BERTopic modeling (this may take a few minutes)...")
topic_model = BERTopic(language="english", calculate_probabilities=True)
sample_texts = df['clean_text'].sample(500, random_state=42).tolist()  # Smaller sample for speed
topics, probs = topic_model.fit_transform(sample_texts)

# Visualize topics
topic_model.visualize_topics()
topic_model.visualize_barchart(top_n_topics=10)
topic_model.visualize_hierarchy()

# ======================
# 3. Medical-Specific Analysis
# ======================

print("\nRunning medical-specific analyses...")

# 1. Medical Claim Detection
print("Detecting medical claims...")
try:
    med7 = pipeline("ner", model="kormilitzin/en_core_med7_lg")
    
    def extract_medical_claims(text):
        try:
            entities = med7(text[:1000])  # Process first 1000 chars for efficiency
            return [ent['word'] for ent in entities if ent['entity_group'] in ['DRUG', 'DOSAGE', 'ADMIN']]
        except:
            return []
    
    df['medical_claims'] = df['text'].apply(extract_medical_claims)
    
    plt.figure(figsize=(10, 6))
    pd.Series([item for sublist in df[df['label']=='fake']['medical_claims'] for item in sublist]).value_counts().head(10).plot(kind='bar', color='salmon', label='Fake')
    pd.Series([item for sublist in df[df['label']=='real']['medical_claims'] for item in sublist]).value_counts().head(10).plot(kind='bar', color='lightblue', alpha=0.7, label='Real')
    plt.title('Top Medical Claims by News Type', fontsize=14)
    plt.xlabel('Medical Term', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
except Exception as e:
    print(f"Could not run medical claim detection: {str(e)}")

# 2. Logical Fallacy Detection (sample)
print("Detecting logical fallacies (sample)...")
try:
    fallacy_detector = pipeline("text-classification", model="vennify/t5-base-fallacy-classification")
    
    def detect_fallacies_sample(text):
        sentences = sent_tokenize(text)[:3]  # First 3 sentences for efficiency
        results = []
        for sent in sentences:
            if len(sent) > 20:
                try:
                    result = fallacy_detector(sent)
                    results.append(result[0]['label'])
                except:
                    continue
        return Counter(results)
    
    sample_df = df.sample(50, random_state=42)
    sample_df['fallacies'] = sample_df['text'].apply(detect_fallacies_sample)
    
    fallacy_df = pd.DataFrame({
        'fake': sum(sample_df[sample_df['label']=='fake']['fallacies'], Counter()),
        'real': sum(sample_df[sample_df['label']=='real']['fallacies'], Counter())
    }).fillna(0)
    
    plt.figure(figsize=(10, 6))
    fallacy_df.plot(kind='bar', ax=plt.gca())
    plt.title('Logical Fallacies in News Sample', fontsize=14)
    plt.xlabel('Fallacy Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.show()
except Exception as e:
    print(f"Could not run fallacy detection: {str(e)}")

# 3. Hedging Language Analysis
print("Analyzing hedging language...")
def detect_hedging(text):
    hedge_words = ['may', 'might', 'could', 'possibly', 'perhaps', 'suggest', 
                  'indicate', 'potential', 'likely', 'unlikely', 'appears']
    words = word_tokenize(text.lower())
    return len([w for w in words if w in hedge_words]) / len(words)

df['hedging_score'] = df['text'].apply(detect_hedging)

plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y='hedging_score', data=df)
plt.title('Hedging Language by News Type', fontsize=14)
plt.xlabel('News Type', fontsize=12)
plt.ylabel('Hedging Score', fontsize=12)
plt.show()

# 4. Evidence-Based Claim Detection
print("Detecting evidence-based claims...")
def detect_evidence(text):
    evidence_phrases = ['studies show', 'research indicates', 'according to', 
                       'clinical trial', 'peer-reviewed', 'meta-analysis',
                       'double-blind', 'randomized controlled']
    text_lower = text.lower()
    return sum(phrase in text_lower for phrase in evidence_phrases)

df['evidence_score'] = df['text'].apply(detect_evidence)

plt.figure(figsize=(10, 6))
sns.countplot(x='evidence_score', hue='label', data=df)
plt.title('Evidence-Based Claims by News Type', fontsize=14)
plt.xlabel('Number of Evidence Phrases', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='News Type')
plt.show()

print("\nAdvanced analysis complete!")
