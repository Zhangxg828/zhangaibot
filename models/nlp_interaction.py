import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import os
from utils.logger import setup_logger

logger = setup_logger("nlp_interaction")


class NLPInteraction:
    def __init__(self, model_path="models/nlp_interaction_model", use_gpu=True):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

        # 使用BlenderBot模型，支持对话生成
        self.model_name = "facebook/blenderbot-400M-distill"
        self.tokenizer = BlenderbotTokenizer.from_pretrained(self.model_name)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(self.model_name)

        # 加载自定义模型（如果存在）
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=self.device))
                logger.info("加载自定义NLP交互模型")
            except Exception as e:
                logger.error(f"加载模型失败: {e}")

        self.model.to(self.device)
        self.model.eval()
        logger.info("自然语言交互模块初始化完成")

    def generate_context(self, trading_data, scoring_data=None, price_data=None, sentiment_data=None):
        """从交易数据和其他模型生成上下文"""
        try:
            context = (
                f"当前代币评分: {trading_data['score']:.2f}, "
                f"价格: {trading_data['price']:.2f}, "
                f"社交情绪: {trading_data['sentiment']:.2f}, "
                f"持有人增长: {trading_data['holder_growth']:.2f}, "
                f"交易决定: {trading_data['decision']['action']}, "
                f"原因: {trading_data['decision'].get('reason', '无特殊原因')}"
            )
            if scoring_data:
                context += f", 评分详情: {scoring_data}"
            if price_data:
                context += f", 预测未来24小时价格: {price_data[:5]}..."  # 只显示前5个预测值
            if sentiment_data:
                context += f", 社交情绪详情: {sentiment_data}"
            return context
        except Exception as e:
            logger.error(f"生成上下文错误: {e}")
            return "无法生成上下文"

    def process_query(self, query, trading_data, scoring_data=None, price_data=None, sentiment_data=None):
        """处理用户自然语言查询"""
        try:
            context = self.generate_context(trading_data, scoring_data, price_data, sentiment_data)
            input_text = f"用户问题: {query}\n上下文: {context}"

            # 编码输入
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    num_beams=5,
                    early_stopping=True
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            logger.info(f"查询: {query}, 回答: {response}")
            return response
        except Exception as e:
            logger.error(f"NLP处理错误: {e}")
            return "抱歉，我无法理解你的问题，请再试一次。"

    def train_model(self, queries, responses, epochs=3, batch_size=8):
        """微调模型"""
        try:
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

            for epoch in range(epochs):
                for i in range(0, len(queries), batch_size):
                    batch_queries = queries[i:i + batch_size]
                    batch_responses = responses[i:i + batch_size]

                    inputs = self.tokenizer(batch_queries, return_tensors="pt", padding=True, truncation=True,
                                            max_length=512)
                    targets = self.tokenizer(batch_responses, return_tensors="pt", padding=True, truncation=True,
                                             max_length=512)

                    inputs = {key: val.to(self.device) for key, val in inputs.items()}
                    targets = targets["input_ids"].to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(**inputs, labels=targets)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)
            logger.info("NLP交互模型微调完成并保存")
        except Exception as e:
            logger.error(f"模型训练错误: {e}")

    def update_model(self, new_queries, new_responses):
        """在线更新模型"""
        try:
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

            inputs = self.tokenizer(new_queries, return_tensors="pt", padding=True, truncation=True, max_length=512)
            targets = self.tokenizer(new_responses, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            targets = targets["input_ids"].to(self.device)

            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=targets)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            self.model.save_pretrained(self.model_path)
            logger.info(f"模型在线更新完成，Loss: {loss.item():.4f}")
        except Exception as e:
            logger.error(f"模型更新错误: {e}")


if __name__ == "__main__":
    # 示例使用
    nlp = NLPInteraction()

    # 模拟训练数据
    train_queries = ["Why buy this token?", "Is it a good investment?"]
    train_responses = ["Because it has a high score.", "Yes, based on current data."]
    nlp.train_model(train_queries, train_responses, epochs=1)

    # 测试交互
    trading_data = {"score": 0.85, "price": 150.0, "sentiment": 0.7, "holder_growth": 0.3,
                    "decision": {"action": "buy", "reason": "高评分"}}
    scoring_data = {"liquidity": 0.8, "sentiment": 0.6}
    price_data = [151, 152, 153, 154, 155]
    sentiment_data = [0.7, -0.2]

    response = nlp.process_query("为什么买入这个代币？", trading_data, scoring_data, price_data, sentiment_data)
    print(f"回答: {response}")