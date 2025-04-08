import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import cv2
import numpy as np
from PIL import Image


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


def extract_image_features(image_path: str) -> np.ndarray:
    """提取图像颜色直方图特征（简化版）"""
    # 读取图像并调整大小
    img = Image.open(image_path).convert('RGB')
    img = img.resize((64, 64))  # 统一尺寸
    img_array = np.array(img)

    # 计算颜色直方图（RGB各通道）
    hist_r = cv2.calcHist([img_array], [0], None, [16], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_array], [1], None, [16], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_array], [2], None, [16], [0, 256]).flatten()
    return np.concatenate([hist_r, hist_g, hist_b])  # 48维特征