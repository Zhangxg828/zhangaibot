CREATE TABLE chain_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    token VARCHAR(50),
    liquidity DECIMAL,
    holder_count INTEGER,
    tx_volume DECIMAL,
    creator_address VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE social_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    source VARCHAR(20),  -- Twitter/Telegram
    text TEXT,
    sentiment_score DECIMAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE trade_records (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    token VARCHAR(50),
    amount DECIMAL,
    price DECIMAL,
    action VARCHAR(20),
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE blacklist (
    address VARCHAR(100) PRIMARY KEY,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);