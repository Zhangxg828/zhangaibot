import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import os
from models.token_scoring import TokenScoring
from models.price_prediction import PricePredictor
from models.sentiment_analysis import SentimentAnalyzer
from utils.logger import setup_logger

logger = setup_logger("nlp_interaction")


class NLPInteraction:
    def __init__(self, model_path="models/nlp_interaction_model", use_gpu=True):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

        self.model_name = "facebook/blenderbot-400M-distill"  # noqa: blenderbot is correct
        self.tokenizer = BlenderbotTokenizer.from_pretrained(self.model_name)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(self.model_name)

        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=self.device))
                logger.info("加载自定义NLP交互模型")
            except Exception as e:
                logger.error(f"加载模型失败: {e}")

        self.model.to(self.device)
        self.model.eval()

        self.token_scorer = TokenScoring()
        self.price_predictor = PricePredictor()
        self.sentiment_analyzer = SentimentAnalyzer()

        logger.info("自然语言交互模块初始化完成")

    @staticmethod
    def generate_context(trade_input, score_input=None, price_input=None, sentiment_input=None):
        """从交易数据和其他模型生成上下文"""
        try:
            context = (
                f"当前代币评分: {trade_input['score']:.2f}, "
                f"价格: {trade_input['price']:.2f}, "
                f"社交情绪: {trade_input['sentiment']:.2f}, "
                f"持有人增长: {trade_input['holder_growth']:.2f}, "
                f"交易决定: {trade_input['decision']['action']}, "
                f"原因: {trade_input['decision'].get('reason', '无特殊原因')}"
            )
            if score_input:
                context += f", 评分详情: {score_input}"
            if price_input:
                context += f", 预测未来24小时价格: {price_input[:5]}..."
            if sentiment_input:
                context += f", 社交情绪详情: {sentiment_input}"
            return context
        except Exception as e:
            logger.error(f"生成上下文错误: {e}")
            return "无法生成上下文"

    def process_query(self, query, trade_input, score_input=None, price_input=None, sentiment_input=None, context=None):
        """处理用户自然语言查询"""
        try:
            if context is None:
                context = self.generate_context(trade_input, score_input, price_input, sentiment_input)
            input_text = f"用户问题: {query}\n上下文: {context}"

            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    num_beams=5,
                    early_stopping=True
                )
            resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            logger.info(f"查询: {query}, 回答: {resp}")
            return resp
        except Exception as e:
            logger.error(f"NLP处理错误: {e}")
            return "抱歉，我无法理解你的问题，请再试一次。"

    def _train_step(self, query_batch, response_batch, optimizer, batch_size=8):
        """提取训练和更新的公共逻辑"""
        if not query_batch or not response_batch:
            logger.error("训练数据为空")
            return torch.tensor(0.0).to(self.device)  # 返回零损失

        loss = torch.tensor(0.0).to(self.device)  # 初始化 loss

        for i in range(0, len(query_batch), batch_size):
            batch_queries = query_batch[i:i + batch_size]
            batch_responses = response_batch[i:i + batch_size]

            inputs = self.tokenizer(batch_queries, return_tensors="pt", padding=True, truncation=True, max_length=512)
            targets = self.tokenizer(batch_responses, return_tensors="pt", padding=True, truncation=True,
                                     max_length=512)

            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            targets = targets["input_ids"].to(self.device)

            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=targets)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        return loss

    def train_model(self, query_batch, response_batch, epochs=3, batch_size=8):
        """微调模型"""
        try:
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)  # 修复parameters引用
            last_loss = 0.0  # 默认值为0.0，避免未赋值

            for epoch in range(epochs):
                last_loss = self._train_step(query_batch, response_batch, optimizer, batch_size)
                logger.info(f"Epoch {epoch}, Loss: {last_loss.item():.4f}")

            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)
            logger.info(f"NLP交互模型微调完成，训练结束时Loss: {last_loss.item():.4f}")
        except Exception as e:
            logger.error(f"模型训练错误: {e}")

    def update_model(self, new_query_batch, new_response_batch, epochs=1, batch_size=8):
        """在线更新模型"""
        try:
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)  # 修复parameters引用
            last_loss = 0.0  # 默认值为0.0，避免未赋值

            for epoch in range(epochs):
                last_loss = self._train_step(new_query_batch, new_response_batch, optimizer, batch_size)
                logger.info(f"Update Epoch {epoch}, Loss: {last_loss.item():.4f}")

            self.model.save_pretrained(self.model_path)
            logger.info(f"模型在线更新完成，更新结束时Loss: {last_loss.item():.4f}")
        except Exception as e:
            logger.error(f"模型更新错误: {e}")


if __name__ == "__main__":
    nlp = NLPInteraction()
    trade_data = {"score": 0.85, "price": 150.0, "sentiment": 0.7, "holder_growth": 0.3,
                  "decision": {"action": "buy", "reason": "高评分"}}
    score_data = {"liquidity": 0.8, "sentiment": 0.6}
    price_data = [151, 152, 153, 154, 155]
    sentiment_data = [0.7, -0.2]

    response = nlp.process_query("为什么买入这个代币？", trade_data, score_data, price_data, sentiment_data)
    print(f"回答: {response}")