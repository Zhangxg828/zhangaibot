import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
from utils.logger import setup_logger

logger = setup_logger("rl_trading")

# DQN网络定义
class DQN(nn.Module):
    def __init__(self, state_size=5, action_size=3):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)  # 动作: 0=买入, 1=卖出, 2=持有

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 交易环境定义
class TradingEnv:
    def __init__(self, price_data):
        self.price_history = price_data
        self.current_step = 0
        self.balance = 1000  # 初始资金
        self.holdings = 0  # 持有代币数量
        self.max_steps = len(self.price_history) - 1

    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.balance = 1000
        self.holdings = 0
        return self.get_state()

    def get_state(self):
        """获取当前状态"""
        price = self.price_history[self.current_step]
        return np.array([price, self.balance, self.holdings, self.current_step / self.max_steps, 0.5])  # [价格, 余额, 持有量, 时间进度, 情绪]

    def step(self, action):
        """执行一步动作"""
        current_price = self.price_history[self.current_step]
        reward = 0
        done = False

        if action == 0:  # 买入
            if self.balance >= current_price:
                self.holdings += 1
                self.balance -= current_price
                reward = 0.1  # 鼓励买入
        elif action == 1:  # 卖出
            if self.holdings > 0:
                self.holdings -= 1
                self.balance += current_price
                reward = (self.balance - 1000) / 1000  # 收益作为奖励
        # action == 2: 持有，无操作

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        next_state = self.get_state()
        return next_state, reward, done

# 强化学习交易器
class RLTrader:
    def __init__(self, state_size=5, action_size=3, model_path="models/rl_trader.pth"):
        """初始化DQN交易器，支持GPU"""
        self.state_size = state_size
        self.action_size = action_size
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 主网络和目标网络
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器和损失函数
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # 经验回放缓冲区
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # 加载预训练模型
        if os.path.exists(model_path):
            try:
                self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
                self.target_net.load_state_dict(self.policy_net.state_dict())
                logger.info(f"加载预训练强化学习模型从: {model_path}")
            except Exception as e:
                logger.error(f"加载模型失败: {e}")

    def decide(self, state):
        """根据状态决定交易动作"""
        try:
            self.policy_net.eval()
            if random.random() < self.epsilon:
                action = random.randint(0, self.action_size - 1)  # 随机探索
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                action = torch.argmax(q_values).item()
            actions = ["buy", "sell", "hold"]
            logger.info(f"状态: {state}, 决策: {actions[action]}")
            return actions[action]
        except Exception as e:
            logger.error(f"决策错误: {e}")
            return "hold"

    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        """训练DQN模型，支持GPU"""
        if len(self.memory) < batch_size:
            return

        try:
            batch = random.sample(self.memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            actions = torch.tensor(actions, dtype=torch.long).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

            # 计算Q值
            self.policy_net.train()
            q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = self.target_net(next_states).max(1)[0].detach()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # 计算损失并优化
            loss = self.criterion(q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 更新探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            logger.debug(f"训练损失: {loss.item()}, 当前探索率: {self.epsilon}")
        except Exception as e:
            logger.error(f"训练错误: {e}")

    def update_target_network(self):
        """更新目标网络"""
        try:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.info("目标网络已更新")
        except Exception as e:
            logger.error(f"更新目标网络失败: {e}")

    def save_model(self):
        """保存模型"""
        try:
            torch.save(self.policy_net.state_dict(), self.model_path)
            logger.info(f"模型保存至 {self.model_path}")
        except Exception as e:
            logger.error(f"保存模型失败: {e}")

    def train_on_history(self, history_prices, episodes=100):
        """基于历史数据训练模型，支持GPU"""
        env = TradingEnv(history_prices)
        try:
            for episode in range(episodes):
                state = env.reset()
                total_reward = 0
                done = False

                while not done:
                    if random.random() < self.epsilon:
                        action = random.randint(0, self.action_size - 1)  # 随机探索
                    else:
                        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                        with torch.no_grad():
                            q_values = self.policy_net(state_tensor)
                        action = torch.argmax(q_values).item()

                    next_state, reward, done = env.step(action)
                    self.store_transition(state, action, reward, next_state, done)
                    self.train(self.batch_size)

                    state = next_state
                    total_reward += reward

                if episode % 10 == 0:
                    self.update_target_network()
                    logger.info(f"回合 {episode}, 总奖励: {total_reward}")

            self.save_model()
        except Exception as e:
            logger.error(f"历史数据训练错误: {e}")

if __name__ == "__main__":
    # 模拟价格历史数据
    price_history = np.random.normal(150, 10, 1000)  # 随机生成1000个价格点
    trader = RLTrader()

    # 训练模型
    trader.train_on_history(price_history, episodes=50)

    # 测试决策
    sample_state = [150.0, 1000, 0, 0.5, 0.7]  # [price, balance, holdings, time_progress, sentiment]
    print(f"决策: {trader.decide(sample_state)}")