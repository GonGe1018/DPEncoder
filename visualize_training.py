import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def load_experiment_data(checkpoint_dir):
    """체크포인트 디렉토리에서 실험 데이터 로드"""
    
    # config.json 읽기
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # train_history.json 읽기
    history_path = os.path.join(checkpoint_dir, "train_history.json")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Training history file not found: {history_path}")
    
    with open(history_path, "r") as f:
        train_history = json.load(f)
    
    return config, train_history

def plot_training_curves(train_history, config, save_path=None):
    """학습 곡선 시각화"""
    
    epochs = train_history["epochs"]
    total_loss = train_history["total_loss"]
    rank_loss = train_history["rank_loss"]
    pairdist_loss = train_history["pairdist_loss"]
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Progress - Experiment ID: {config["experiment_id"][:8]}...', fontsize=16)
    
    # Total Loss
    axes[0, 0].plot(epochs, total_loss, 'b-', linewidth=2, label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Rank Loss
    axes[0, 1].plot(epochs, rank_loss, 'r-', linewidth=2, label='Rank Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Rank Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Pair Distance Loss
    axes[1, 0].plot(epochs, pairdist_loss, 'g-', linewidth=2, label='Pair Distance Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Pair Distance Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # All losses combined
    axes[1, 1].plot(epochs, total_loss, 'b-', linewidth=2, label='Total Loss')
    axes[1, 1].plot(epochs, rank_loss, 'r-', linewidth=2, label='Rank Loss')
    axes[1, 1].plot(epochs, pairdist_loss, 'g-', linewidth=2, label='Pair Distance Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('All Losses Combined')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def print_experiment_summary(config, train_history):
    """실험 요약 정보 출력"""
    
    print("=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Experiment ID: {config['experiment_id']}")
    print(f"Timestamp: {config['timestamp']}")
    print(f"Device: {config['device']}")
    print()
    print("Hyperparameters:")
    print(f"  - Batch Size: {config['batch_size']}")
    print(f"  - Hidden Dims: {config['hidden_dims']}")
    print(f"  - Latent Dim: {config['latent_dim']}")
    print(f"  - k: {config['k']}")
    print(f"  - Lambda Rank: {config['lambda_rank']}")
    print(f"  - Lambda Pairdist: {config['lambda_pairdist']}")
    print(f"  - Learning Rate: {config['lr']}")
    print(f"  - Epochs: {config['epochs']}")
    print()
    
    if train_history["epochs"]:
        final_epoch = train_history["epochs"][-1]
        final_total = train_history["total_loss"][-1]
        final_rank = train_history["rank_loss"][-1]
        final_pair = train_history["pairdist_loss"][-1]
        
        print("Final Results:")
        print(f"  - Final Epoch: {final_epoch}")
        print(f"  - Final Total Loss: {final_total:.4f}")
        print(f"  - Final Rank Loss: {final_rank:.4f}")
        print(f"  - Final Pair Distance Loss: {final_pair:.4f}")
        
        # 최적 성능 찾기
        best_epoch = np.argmin(train_history["total_loss"]) + 1
        best_total = min(train_history["total_loss"])
        
        print(f"  - Best Total Loss: {best_total:.4f} (Epoch {best_epoch})")
    print("=" * 60)

def list_experiments():
    """사용 가능한 실험 목록 출력"""
    
    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        print("No checkpoints directory found.")
        return []
    
    experiment_dirs = [d for d in os.listdir(checkpoints_dir) 
                      if os.path.isdir(os.path.join(checkpoints_dir, d))]
    
    if not experiment_dirs:
        print("No experiments found.")
        return []
    
    print("Available experiments:")
    valid_experiments = []
    
    for i, exp_dir in enumerate(experiment_dirs):
        exp_path = os.path.join(checkpoints_dir, exp_dir)
        config_path = os.path.join(exp_path, "config.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                print(f"  {i+1}. {exp_dir[:8]}... ({config.get('timestamp', 'Unknown time')})")
                valid_experiments.append(exp_dir)
            except:
                print(f"  {i+1}. {exp_dir} (Invalid config)")
        else:
            print(f"  {i+1}. {exp_dir} (No config)")
    
    return valid_experiments

def main():
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("--experiment_id", type=str, help="Experiment ID (UUID) to visualize")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--save", type=str, help="Path to save the plot")
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    if args.experiment_id:
        checkpoint_dir = os.path.join("checkpoints", args.experiment_id)
    else:
        # 대화형 모드: 사용자가 실험 선택
        experiments = list_experiments()
        if not experiments:
            return
        
        try:
            choice = int(input(f"\nSelect experiment (1-{len(experiments)}): ")) - 1
            if 0 <= choice < len(experiments):
                checkpoint_dir = os.path.join("checkpoints", experiments[choice])
            else:
                print("Invalid choice.")
                return
        except ValueError:
            print("Invalid input.")
            return
    
    try:
        config, train_history = load_experiment_data(checkpoint_dir)
        print_experiment_summary(config, train_history)
        
        # 저장 경로 설정
        save_path = args.save
        if not save_path:
            save_path = os.path.join(checkpoint_dir, "training_curves.png")
        
        plot_training_curves(train_history, config, save_path)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
