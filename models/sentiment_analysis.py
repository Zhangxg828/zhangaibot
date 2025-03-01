import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import os
from utils.logger import setup_logger

logger = setup_logger("sentiment_analysis")

class SentimentAnalyzer:
    def __init__(self, model_path="models/sentiment_model", use_gpu=True):
        """初始化情绪分析模型，增强GPU支持"""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

        # 使用多语言BERT模型
        self.model_name = "nlptown/bert-base-multilingual-uncased-sentiment"  # noqa: nlptown is correct
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name).to(self.device)

        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=self.device))
                logger.info(f"加载微调情绪分析模型从: {model_path}")
            except Exception as e:
                logger.error(f"加载模型失败: {e}")

        self.model.eval()

        # 初始化情感词典
        self.sentiment_dict = {
            "great": 0.8, "good": 0.5, "awesome": 0.7,
            "bad": -0.5, "scam": -0.8, "shit": -0.7
        }
        logger.info("情绪分析模块初始化完成")

    def preprocess_text(self, _prep_texts):
        """预处理文本"""
        if isinstance(_prep_texts, str):
            _prep_texts_list = [_prep_texts]
        else:
            _prep_texts_list = _prep_texts
        try:
            inputs = self.tokenizer(_prep_texts_list, padding=True, truncation=True, max_length=128,
                                    return_tensors="pt")
            return {key: val.to(self.device) for key, val in inputs.items()}
        except Exception as e:
            logger.error(f"文本预处理错误: {e}")
            return None

    def analyze_dict(self, text):
        """基于情感词典分析"""
        words = text.lower().split()
        _dict_sentiment_scores = [self.sentiment_dict.get(word, 0.0) for word in words]
        return np.mean(_dict_sentiment_scores) if _dict_sentiment_scores else 0.0

    def analyze(self, _analyze_texts, batch_size=32, source="twitter"):
        """分析文本情绪，支持批量处理和GPU"""
        try:
            if isinstance(_analyze_texts, str):
                _analyze_texts = [_analyze_texts]

            all_scores = []
            for i in range(0, len(_analyze_texts), batch_size):
                _batch_analyze_texts = _analyze_texts[i:i + batch_size]
                inputs = self.preprocess_text(_batch_analyze_texts)
                if inputs is None:
                    return [0.0] * len(_analyze_texts)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    _batch_analyze_scores = probs[:, 2] - probs[:, 0]  # 正向-负向，范围-1到1

                dict_scores = [self.analyze_dict(text) for text in _batch_analyze_texts]
                final_scores = [(s.item() + d) / 2 for s, d in zip(_batch_analyze_scores, dict_scores)]
                all_scores.extend(final_scores)

            avg_score = np.mean(all_scores) if all_scores else 0.0
            logger.info(f"分析 {len(_analyze_texts)} 条 {source} 文本，平均情绪得分: {avg_score:.3f}")
            return all_scores if len(_analyze_texts) > 1 else all_scores[0]
        except Exception as e:
            logger.error(f"情绪分析错误: {e}")
            return [0.0] * len(_analyze_texts) if isinstance(_analyze_texts, list) else 0.0

    def train_model(self, input_texts_train, input_labels_train, epochs=3, batch_size=16, validation_split=0.2):
        """微调模型，支持GPU和批量训练"""
        try:
            from sklearn.model_selection import train_test_split
            texts_train, val_texts, labels_train, val_labels = train_test_split(
                input_texts_train, input_labels_train, test_size=validation_split, random_state=42
            )

            # 预处理训练和验证数据
            train_inputs = self.preprocess_text(texts_train)
            val_inputs = self.preprocess_text(val_texts)
            if train_inputs is None or val_inputs is None:
                raise ValueError("预处理失败")
            labels_train_tensor = torch.tensor(labels_train, dtype=torch.long).to(self.device)
            val_labels = torch.tensor(val_labels, dtype=torch.long).to(self.device)

            # 创建DataLoader
            train_dataset = torch.utils.data.TensorDataset(
                train_inputs["input_ids"], train_inputs["attention_mask"], labels_train_tensor
            )
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
            self.model.train()
            last_loss = 0.0

            for epoch in range(epochs):
                total_loss = 0
                for batch in train_loader:
                    input_ids, attention_mask, batch_labels = [b.to(self.device) for b in batch]
                    optimizer.zero_grad()
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
                    last_loss = outputs.loss
                    last_loss.backward()
                    optimizer.step()
                    total_loss += last_loss.item()

                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(**val_inputs, labels=val_labels)
                    val_loss = val_outputs.loss
                logger.info(f"Epoch {epoch}, Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss.item():.4f}")
                self.model.train()

            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)
            logger.info(f"情绪分析模型微调完成并保存至: {self.model_path}")
        except Exception as e:
            logger.error(f"模型训练错误: {e}")

    def update_model(self, new_texts, new_labels, epochs=1, batch_size=16):
        """在线更新模型，支持GPU"""
        try:
            inputs = self.preprocess_text(new_texts)
            if inputs is None:
                raise ValueError("预处理失败")
            labels = torch.tensor(new_labels, dtype=torch.long).to(self.device)

            dataset = torch.utils.data.TensorDataset(inputs["input_ids"], inputs["attention_mask"], labels)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
            self.model.train()

            for epoch in range(epochs):
                total_loss = 0
                for batch in loader:
                    input_ids, attention_mask, batch_labels = [b.to(self.device) for b in batch]
                    optimizer.zero_grad()
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                logger.info(f"Update Epoch {epoch}, Loss: {total_loss / len(loader):.4f}")

            self.model.save_pretrained(self.model_path)
            logger.info(f"模型在线更新完成并保存至: {self.model_path}")
        except Exception as e:
            logger.error(f"模型更新错误: {e}")

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    train_texts = ["Great coin!", "Terrible scam", "Awesome project"]
    train_labels = [2, 0, 2]  # 0=负向, 1=中性, 2=正向
    analyzer.train_model(train_texts, train_labels, epochs=1)

    texts = ["This is a great coin!", "Total scam, avoid!"]
    scores = analyzer.analyze(texts)
    print(f"情绪得分: {scores}")