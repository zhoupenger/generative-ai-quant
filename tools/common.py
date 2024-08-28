import pandas as pd

def generate_simple_quant_chain_database(path):
    df = pd.DataFrame({
        "signal": ["Test"], 
        "train_return_rate": [0],
        "train_max_drawdown": [0],	
        "train_sharpe_ratio" : [0],
        "test_return_rate": [0],
        "test_max_drawdown": [0],	
        "test_sharpe_ratio": [0]
    })

    df.to_feather(path)

def update_database(new_df, path):
    original_df = pd.read_feather(path)
    
    new_df = new_df.reset_index()
    new_df.rename(columns={'index': 'signal'}, inplace=True)

    # 连接原始 DataFrame 和新 DataFrame 
    updated_df = pd.concat([original_df, new_df], ignore_index=True)

    updated_df.to_feather(path)