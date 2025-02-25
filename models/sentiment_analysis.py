import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import numpy as np
from collections import defaultdict
import os
from utils.logger import setup_logger

logger = setup_logger("sentiment_analysis")


class SentimentAnalyzer:
    def __init__(self, model_path="models/sentiment_model", use_gpu=True):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

        # 使用多语言BERT模型
        self.model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)

        # 加载微调模型（如果存在）
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=self.device))
                logger.info("加载微调情绪分析模型")
            except Exception as e:
                logger.error(f"加载模型失败: {e}")

        self.model.to(self.device)
        self.model.eval()

        # 初始化情感词典（简单示例，可替换为专业词典）
        self.sentiment_dict = {
            "great": 0.8, "good": 0.5, "awesome": 0.7,
            "bad": -0.5, "scam": -0.8, "shit": -0.7
        }
        logger.info("情绪分析模块初始化完成")

    def preprocess_text(self, texts):
        """预处理文本"""
        if isinstance(texts, str):
            texts = [texts]
        try:
            inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            return inputs
        except Exception as e:
            logger.error(f"文本预处理错误: {e}")
            return None

    def analyze_dict(self, text):
        """基于情感词典分析"""
        words = text.lower().split()
        scores = [self.sentiment_dict.get(word, 0.0) for word in words]
        return np.mean(scores) if scores else 0.0

    def analyze(self, texts, batch_size=32):
        """分析文本情绪（支持批量）"""
        try:
            if isinstance(texts, str):
                texts = [texts]

            # 分批处理
            all_scores = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.preprocess_text(batch_texts)
                if inputs is None:
                    return [0.0] * len(texts)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)

                    # 假设模型输出3类：负向(0)、中性(1)、正向(2)
                    scores = probs[:, 2] - probs[:, 0]  # 正向概率 - 负向概率，范围-1到1

                # 结合情感词典
                dict_scores = [self.analyze_dict(text) for text in batch_texts]
                final_scores = [(s.item() + d) / 2 for s, d in zip(scores, dict_scores)]
                all_scores.extend(final_scores)

            logger.info(f"分析 {len(texts)} 条文本，平均情绪得分: {np.mean(all_scores):.3f}")
            return all_scores if len(texts) > 1 else all_scores[0]
        except Exception as e:
            logger.error(f"情绪分析错误: {e}")
            return [0.0] * len(texts) if isinstance(texts, list) else 0.0

    def train_model(self, texts, labels, epochs=3, batch_size=16, validation_split=0.2):
        """微调模型"""
        try:
            from sklearn.model_selection import train_test_split
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=validation_split, random_state=42
            )

            # 准备数据
            train_inputs = self.preprocess_text(train_texts)
            val_inputs = self.preprocess_text(val_texts)
            train_labels = torch.tensor(train_labels, dtype=torch.long).to(self.device)
            val_labels = torch.tensor(val_labels, dtype=torch.long).to(self.device)

            # 训练设置
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
            self.model.train()

            for epoch in range(epochs):
                for i in range(0, len(train_texts), batch_size):
                    batch_inputs = {k: v[i:i + batch_size] for k, v in train_inputs.items()}
                    batch_labels = train_labels[i:i + batch_size]

                    optimizer.zero_grad()
                    outputs = self.model(**batch_inputs, labels=batch_labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                # 验证
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(**val_inputs, labels=val_labels)
                    val_loss = val_outputs.loss
                logger.info(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
                self.model.train()

            # 保存模型
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)
            logger.info("情绪分析模型微调完成并保存")
        except Exception as e:
            logger.error(f"模型训练错误: {e}")

    def update_model(self, new_texts, new_labels, epochs=1):
        """在线更新模型"""
        try:
            self.model.train()
            inputs = self.preprocess_text(new_texts)
            labels = torch.tensor(new_labels, dtype=torch.long).to(self.device)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                logger.info(f"Update Epoch {epoch}, Loss: {loss.item():.4f}")

            self.model.save_pretrained(self.model_path)
            logger.info("模型在线更新完成")
        except Exception as e:
            logger.error(f"模型更新错误: {e}")


if __name__ == "__main__":
    # 示例使用
    analyzer = SentimentAnalyzer()

    # 模拟训练数据
    train_texts = ["Great coin!", "Terrible scam", "Awesome project"]
    train_labels = [2, 0, 2]  # 0=负向, 1=中性, 2=正向
    analyzer.train_model(train_texts, train_labels, epochs=1)

    # 分析示例
    texts = ["This is a great coin!", "Total scam, avoid!"]
    scores = analyzer.analyze(texts)
    print(f"情绪得分: {scores}")