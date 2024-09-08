import os
import glob
from datetime import datetime
import torch

def get_latest_model(model_dir):
    # 获取模型目录中所有以 .pth 结尾的文件
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in directory: {model_dir}")
    
    # 按文件名中的时间戳进行排序
    model_files.sort(key=lambda f: datetime.strptime(f, 'model_%Y%m%d_%H%M%S.pth'))
    
    # 返回最新的模型文件
    latest_model = model_files[-1]
    
    return os.path.join(model_dir, latest_model)

def save_model(model, model_dir):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'model_{current_time}.pth'
    torch.save(model.state_dict(), os.path.join(model_dir, model_filename))

# print(get_latest_model('models'))