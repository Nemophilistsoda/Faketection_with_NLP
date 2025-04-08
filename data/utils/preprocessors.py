import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_path):
    df = pd.read_csv(file_path)
    # 需要确认实际数据集的列名
    df = df.dropna(subset=['text', 'label'])  # 修改为实际存在的列名
    return df

def extract_tfidf_features(df: pd.DataFrame, max_features: int = 1000) -> tuple:
    """提取TF-IDF特征"""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df['text'])
    return X, df['label'], vectorizer