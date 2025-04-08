import os
import pandas as pd
from tqdm import tqdm
from data.utils.preprocessors import extract_image_features

def process_images(data_dir: str, df: pd.DataFrame) -> pd.DataFrame:
    """为数据集中的每条文本添加对应的图像特征"""
    image_features = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(data_dir, f"{row['id']}.jpg")  # 假设图像名按id命名
        features = extract_image_features(image_path)
        image_features.append(features)
    df['image_features'] = image_features
    return df