import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from data.utils.preprocessors import load_data, extract_tfidf_features
from .image_processor import process_images


def main():
    # 加载文本数据
    df = load_data('data/raw/fake_news.csv')

    # 提取文本特征
    X_text, y, vectorizer = extract_tfidf_features(df)

    # 提取图像特征（假设图像在data/raw/images目录下）
    df = process_images('data/raw/images', df)
    X_image = np.stack(df['image_features'].values)

    # 特征拼接
    X_combined = np.concatenate([X_text.toarray(), X_image], axis=1)

    # 训练分类器
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    print(f"多模态准确率: {clf.score(X_test, y_test):.2f}")


if __name__ == "__main__":
    main()