import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys
import os

# 添加项目根目录到Python路径（确保模块引用正确）
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.utils.preprocessors import load_data, extract_tfidf_features


def main():
    # 1. 加载数据
    # 修改为动态获取项目根目录路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, 'data', 'raw', 'WELFake_Dataset.csv')

    df = load_data(data_path)

    # 2. 提取特征
    X, y, vectorizer = extract_tfidf_features(df)

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 训练模型
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # 5. 评估
    y_pred = clf.predict(X_test)
    print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")


if __name__ == "__main__":
    main()