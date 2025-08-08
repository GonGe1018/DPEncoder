#!/usr/bin/env python3
"""
간단한 학습 결과 시각화 스크립트
최신 실험을 자동으로 찾아서 시각화합니다.
"""

import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

def find_latest_experiment():
    """가장 최근 실험 찾기"""
    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        return None
    
    experiment_dirs = [d for d in os.listdir(checkpoints_dir) 
                      if os.path.isdir(os.path.join(checkpoints_dir, d))]
    
    latest_time = None
    latest_exp = None
    
    for exp_dir in experiment_dirs:
        config_path = os.path.join(checkpoints_dir, exp_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                timestamp = datetime.fromisoformat(config.get('timestamp', ''))
                if latest_time is None or timestamp > latest_time:
                    latest_time = timestamp
                    latest_exp = exp_dir
            except:
                continue
    
    return latest_exp

def quick_plot():
    """빠른 시각화"""
    latest_exp = find_latest_experiment()
    
    if not latest_exp:
        print("No experiments found!")
        return
    
    checkpoint_dir = os.path.join("checkpoints", latest_exp)
    
    # 데이터 로드
    with open(os.path.join(checkpoint_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    with open(os.path.join(checkpoint_dir, "train_history.json"), "r") as f:
        train_history = json.load(f)
    
    # 간단한 플롯
    plt.figure(figsize=(12, 4))
    
    epochs = train_history["epochs"]
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_history["total_loss"], 'b-', linewidth=2)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_history["rank_loss"], 'r-', linewidth=2)
    plt.title('Rank Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_history["pairdist_loss"], 'g-', linewidth=2)
    plt.title('Pair Distance Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Latest Training Results - {latest_exp[:8]}...', fontsize=14)
    plt.tight_layout()
    
    # 저장
    save_path = os.path.join(checkpoint_dir, "quick_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Latest experiment: {latest_exp}")
    print(f"Plot saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    quick_plot()
