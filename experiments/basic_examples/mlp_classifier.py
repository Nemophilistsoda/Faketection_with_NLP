# experiments/basic_examples/mlp_classifier.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data.utils.preprocessors import load_data, extract_tfidf_features


class TextClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def main():
    # 加载数据与特征
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, 'data', 'raw', 'WELFake_Dataset.csv')
    df = load_data(data_path)

    X, y, _ = extract_tfidf_features(df, max_features=1000)

    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X.toarray())
    y_tensor = torch.FloatTensor(y.values)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 训练模型
    model = TextClassifier(input_dim=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    # 新增评估代码
    with torch.no_grad():
        model.eval()
        test_outputs = model(X_tensor)
        predicted = (test_outputs.squeeze() > 0.5).float()
        accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
        print(f"\n最终测试准确率: {accuracy:.2%}")

# 添加程序入口（关键缺失）
if __name__ == "__main__":
    main()