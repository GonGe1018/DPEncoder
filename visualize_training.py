import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta

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

def plot_training_curves(train_history, config, save_path=None, recall=None, recall_history=None):
    """학습 곡선 시각화"""
    
    epochs = train_history["epochs"]
    total_loss = train_history["total_loss"]
    rank_loss = train_history["rank_loss"]
    pairdist_loss = train_history["pairdist_loss"]
    
    # Loss 함수 정보 추가
    loss_function = config.get("loss_function", "Unknown")
    tau = config.get("tau", "N/A")
    
    # 그래프 생성 - recall이 있으면 2x3, 없으면 2x2
    if recall is not None or recall_history is not None:
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    title_suffix = f' (Loss: {loss_function}'
    if "V2" in loss_function:
        title_suffix += f', τ={tau}'
    if recall is not None:
        title_suffix += f', Final Recall@{config.get("k", 5)}={recall:.4f}'
    title_suffix += ')'
    
    fig.suptitle(f'Training Progress - {config["experiment_id"][:8]}...{title_suffix}', fontsize=16)
    
    # Total Loss (항상 로그 스케일)
    axes[0, 0].semilogy(epochs, total_loss, 'b-', linewidth=2, label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (log scale)')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Rank Loss (로그 스케일로 변경)
    axes[0, 1].semilogy(epochs, rank_loss, 'r-', linewidth=2, label='Rank Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss (log scale)')
    axes[0, 1].set_title('Rank Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Pair Distance Loss (항상 로그 스케일)
    axes[1, 0].semilogy(epochs, pairdist_loss, 'g-', linewidth=2, label='Pair Distance Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss (log scale)')
    axes[1, 0].set_title('Pair Distance Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # All losses combined (로그 스케일)
    axes[1, 1].semilogy(epochs, total_loss, 'b-', linewidth=2, label='Total Loss')
    axes[1, 1].semilogy(epochs, rank_loss, 'r-', linewidth=2, label='Rank Loss')
    axes[1, 1].semilogy(epochs, pairdist_loss, 'g-', linewidth=2, label='Pair Distance Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss (log scale)')
    axes[1, 1].set_title('All Losses Combined')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    if recall is not None or recall_history is not None:
        # Recall 히스토리가 있으면 그래프로, 없으면 텍스트로
        if recall_history is not None and len(recall_history[0]) > 0:
            recall_epochs, recall_values = recall_history
            axes[0, 2].plot(recall_epochs, recall_values, 'purple', marker='o', linewidth=2, markersize=6)
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Recall Score')
            axes[0, 2].set_title(f'KNN Recall@{config.get("k", 5)} History')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 최종 recall 값 표시
            if len(recall_values) > 0:
                final_recall = recall_values[-1]
                axes[0, 2].text(0.05, 0.95, f'Final: {final_recall:.4f}', 
                               transform=axes[0, 2].transAxes, fontsize=12, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        else:
            # Recall 히스토리가 없으면 최종 recall 값만 표시
            recall_text = f'KNN Recall@{config.get("k", 5)}\n\n{recall:.4f}' if recall is not None else 'Recall\nNot Available'
            axes[0, 2].text(0.5, 0.5, recall_text, 
                           horizontalalignment='center', verticalalignment='center', 
                           transform=axes[0, 2].transAxes, fontsize=24, fontweight='bold')
            axes[0, 2].set_title('Final Recall Score')
            axes[0, 2].axis('off')
        
        # 성능 요약 (시간 정보 포함)
        final_total = total_loss[-1]
        best_total = min(total_loss)
        best_epoch = np.argmin(total_loss) + 1
        
        summary_text = f'Final Loss: {final_total:.4f}\nBest Loss: {best_total:.4f}\n(Epoch {best_epoch})'
        if recall is not None:
            summary_text += f'\nFinal Recall: {recall:.4f}'
        
        # 시간 정보 추가
        summary_text += '\n' + '='*20 + '\nTiming Info:\n'
        if 'time_statistics' in train_history:
            stats = train_history['time_statistics']
            summary_text += f'Avg: {stats["average_epoch_time"]:.2f}s/epoch\n'
            summary_text += f'Range: {stats["min_epoch_time"]:.2f}s - {stats["max_epoch_time"]:.2f}s\n'
        elif 'epoch_times' in train_history and len(train_history['epoch_times']) > 0:
            epoch_times = train_history['epoch_times']
            avg_time = sum(epoch_times) / len(epoch_times)
            min_time = min(epoch_times)
            max_time = max(epoch_times)
            summary_text += f'Avg: {avg_time:.2f}s/epoch\n'
            summary_text += f'Range: {min_time:.2f}s - {max_time:.2f}s\n'
        
        if 'cumulative_time' in train_history and len(train_history['cumulative_time']) > 0:
            total_seconds = train_history['cumulative_time'][-1]
            total_time_str = str(timedelta(seconds=int(total_seconds)))
            summary_text += f'Total: {total_time_str}'
        
        axes[1, 2].text(0.5, 0.5, summary_text, 
                       horizontalalignment='center', verticalalignment='center', 
                       transform=axes[1, 2].transAxes, fontsize=12, fontfamily='monospace')
        axes[1, 2].set_title('Performance & Timing Summary')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def print_experiment_summary(config, train_history, recall=None):
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
    print(f"  - Loss Function: {config.get('loss_function', 'Unknown')}")
    if config.get('tau'):
        print(f"  - Tau (temperature): {config['tau']}")
    print()
    
    # 시간 정보 표시
    if 'epoch_times' in train_history and len(train_history['epoch_times']) > 0:
        print("Training Time Information:")
        
        # 총 학습 시간 계산
        if 'total_training_time_formatted' in train_history:
            print(f"  - Total Training Time: {train_history['total_training_time_formatted']}")
        elif 'cumulative_time' in train_history and len(train_history['cumulative_time']) > 0:
            total_seconds = train_history['cumulative_time'][-1]
            total_time_str = str(timedelta(seconds=int(total_seconds)))
            print(f"  - Total Training Time: {total_time_str}")
        
        # 시간 통계
        if 'time_statistics' in train_history:
            stats = train_history['time_statistics']
            print(f"  - Average Time per Epoch: {stats['average_epoch_time']:.2f}s")
            print(f"  - Fastest Epoch: {stats['min_epoch_time']:.2f}s")
            print(f"  - Slowest Epoch: {stats['max_epoch_time']:.2f}s")
        elif 'epoch_times' in train_history:
            epoch_times = train_history['epoch_times']
            avg_time = sum(epoch_times) / len(epoch_times)
            min_time = min(epoch_times)
            max_time = max(epoch_times)
            print(f"  - Average Time per Epoch: {avg_time:.2f}s")
            print(f"  - Fastest Epoch: {min_time:.2f}s")
            print(f"  - Slowest Epoch: {max_time:.2f}s")
        
        # 시작/종료 시간
        if 'training_start_time' in train_history:
            print(f"  - Training Start: {train_history['training_start_time']}")
        if 'training_end_time' in train_history:
            print(f"  - Training End: {train_history['training_end_time']}")
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
        
        if recall is not None:
            print(f"  - KNN Recall@{config.get('k', 5)}: {recall:.4f}")
        
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
    valid_index = 1
    
    for exp_dir in experiment_dirs:
        exp_path = os.path.join(checkpoints_dir, exp_dir)
        config_path = os.path.join(exp_path, "config.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                loss_func = config.get('loss_function', 'Unknown')
                timestamp = config.get('timestamp', 'Unknown time')
                print(f"  {valid_index}. {exp_dir[:8]}... ({timestamp[:19]}) - Loss: {loss_func}")
                valid_experiments.append(exp_dir)
                valid_index += 1
            except:
                pass  # 무효한 config는 출력하지 않음
    
    if not valid_experiments:
        print("No valid experiments found.")
    
    return valid_experiments

def main():
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("--experiment_id", type=str, help="Experiment ID (UUID) to visualize")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--save", type=str, help="Path to save the plot")
    parser.add_argument("--compare", action="store_true", help="Compare multiple experiments")
    parser.add_argument("--no-recall", action="store_true", help="Skip recall display (faster)")
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    if args.compare:
        compare_experiments(skip_recall=args.no_recall)
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
        
        # 저장된 recall history 사용 (계산하지 않음)
        recall = None
        recall_history = None
        
        if not args.no_recall:
            # 저장된 recall history가 있는지 확인
            if 'recall_history' in train_history and len(train_history['recall_history']) > 0:
                print("Using saved recall history from training...")
                recall_epochs = train_history['recall_epochs']
                recall_values = train_history['recall_history']
                recall_history = (recall_epochs, recall_values)
                recall = recall_values[-1]  # 마지막 recall 값 사용
            else:
                print("No saved recall history found. Train with newer version to get recall data.")
        
        print_experiment_summary(config, train_history, recall)
        
        # 저장 경로 설정
        save_path = args.save
        if not save_path:
            save_path = os.path.join(checkpoint_dir, "training_curves.png")
        
        plot_training_curves(train_history, config, save_path, recall, recall_history)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def compare_experiments(skip_recall=False):
    """여러 실험을 비교하는 함수"""
    experiments = list_experiments()
    if not experiments:
        return
    
    print(f"\nSelect experiments to compare (comma separated, e.g., 1,2,3):")
    try:
        choices_input = input("Experiment numbers: ")
        choices = [int(x.strip()) - 1 for x in choices_input.split(',')]
        
        if any(choice < 0 or choice >= len(experiments) for choice in choices):
            print("Invalid choice(s).")
            return
        
        selected_experiments = [experiments[choice] for choice in choices]
        
        # 데이터 로드
        experiment_data = []
        for exp_id in selected_experiments:
            checkpoint_dir = os.path.join("checkpoints", exp_id)
            try:
                config, train_history = load_experiment_data(checkpoint_dir)
                
                # Recall 계산 (skip_recall=False인 경우)
                recall = None
                recall_hist = None
                
                if not skip_recall:
                    # 저장된 recall history가 있는지 확인
                    if 'recall_history' in train_history and len(train_history['recall_history']) > 0:
                        print(f"Using saved recall history for {exp_id[:8]}...")
                        recall_epochs = train_history['recall_epochs']
                        recall_values = train_history['recall_history']
                        recall_hist = (recall_epochs, recall_values)
                        recall = recall_values[-1]  # 마지막 recall 값 사용
                    else:
                        print(f"No saved recall history found for {exp_id[:8]}...")
                        recall = None
                        recall_hist = None
                
                experiment_data.append((exp_id, config, train_history, recall, recall_hist))
            except Exception as e:
                print(f"Error loading {exp_id}: {e}")
        
        if not experiment_data:
            print("No valid experiments to compare.")
            return
        
        # 비교 그래프 생성
        plot_comparison(experiment_data)
        
    except ValueError:
        print("Invalid input format.")
    except Exception as e:
        print(f"Error: {e}")

def plot_comparison(experiment_data):
    """여러 실험의 비교 그래프 생성"""
    
    # Recall 정보가 있는지 확인
    has_recall = any(len(data) > 3 and data[3] is not None for data in experiment_data)
    has_recall_history = any(len(data) > 4 and data[4] is not None and len(data[4][0]) > 0 for data in experiment_data)
    
    if has_recall or has_recall_history:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    fig.suptitle('Experiment Comparison', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, data in enumerate(experiment_data):
        color = colors[i % len(colors)]
        
        # 데이터 언패킹 (길이에 따라 다르게 처리)
        if len(data) >= 5:
            exp_id, config, train_history, recall, recall_hist = data
        elif len(data) >= 4:
            exp_id, config, train_history, recall = data
            recall_hist = None
        else:
            exp_id, config, train_history = data
            recall = None
            recall_hist = None
        
        loss_func = config.get('loss_function', 'Unknown')
        label = f"{exp_id[:8]}... ({loss_func})"
        
        epochs = train_history["epochs"]
        total_loss = train_history["total_loss"]
        rank_loss = train_history["rank_loss"]
        pairdist_loss = train_history["pairdist_loss"]
        
        # Total Loss (항상 로그 스케일)
        axes[0, 0].semilogy(epochs, total_loss, color=color, linewidth=2, label=label)
        
        # Rank Loss (로그 스케일로 변경)
        axes[0, 1].semilogy(epochs, rank_loss, color=color, linewidth=2, label=label)
        
        # Pair Distance Loss (항상 로그 스케일)
        axes[1, 0].semilogy(epochs, pairdist_loss, color=color, linewidth=2, label=label)
        
        # KNN Recall History or Final Recall
        if recall_hist is not None and len(recall_hist[0]) > 0:
            # 에포크별 recall history가 있는 경우
            recall_epochs, recall_values = recall_hist
            axes[1, 1].plot(recall_epochs, recall_values, color=color, marker='o', linewidth=2, markersize=4, label=label)
        elif recall is not None:
            # 최종 recall만 있는 경우 - 마지막 에포크에 점으로 표시
            final_epoch = epochs[-1] if epochs else 1
            axes[1, 1].scatter([final_epoch], [recall], color=color, s=100, marker='*', label=f"{label} (Final)")
            # 실험명과 함께 recall 값 표시
            axes[1, 1].annotate(f'{recall:.3f}', 
                               xy=(final_epoch, recall), 
                               xytext=(5, 5), 
                               textcoords='offset points',
                               fontsize=8, 
                               color=color)
        
        # Recall history는 이미 axes[1, 1]에서 처리됨
    
    # 그래프 설정
    axes[0, 0].set_title('Total Loss Comparison (Log Scale)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (log scale)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_title('Rank Loss Comparison (Log Scale)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss (log scale)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].set_title('Pair Distance Loss Comparison (Log Scale)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss (log scale)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].set_title('KNN Recall@5 History Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall Score')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Recall 정보가 있는 경우 추가 플롯
    if has_recall or has_recall_history:
        # Recall 비교 바 차트
        exp_names = [f"{data[0][:8]}...\n({data[1].get('loss_function', 'Unknown')})" 
                    for data in experiment_data]
        recall_values = []
        for data in experiment_data:
            if len(data) > 4 and data[4] is not None and len(data[4][1]) > 0:
                recall_values.append(data[4][1][-1])  # 마지막 recall 값
            elif len(data) > 3 and data[3] is not None:
                recall_values.append(data[3])  # 최종 recall 값
            else:
                recall_values.append(0)
        
        bars = axes[0, 2].bar(range(len(exp_names)), recall_values, color=colors[:len(experiment_data)])
        axes[0, 2].set_title('Final Recall Comparison')
        axes[0, 2].set_ylabel('Recall Score')
        axes[0, 2].set_xticks(range(len(exp_names)))
        axes[0, 2].set_xticklabels(exp_names, rotation=45, ha='right')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # 바 위에 값 표시
        for bar, recall in zip(bars, recall_values):
            if recall > 0:
                axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{recall:.3f}', ha='center', va='bottom')
        
        # 성능 요약 테이블 (시간 정보 포함)
        axes[1, 2].axis('off')
        summary_text = "Performance & Timing Summary:\n\n"
        for data in experiment_data:
            exp_id, config, train_history = data[:3]
            recall = data[3] if len(data) > 3 else None
            
            loss_func = config.get('loss_function', 'Unknown')
            final_loss = train_history["total_loss"][-1]
            best_loss = min(train_history["total_loss"])
            recall_str = f"{recall:.4f}" if recall is not None else "N/A"
            
            summary_text += f"{exp_id[:8]}... ({loss_func}):\n"
            summary_text += f"  Final: {final_loss:.4f}\n"
            summary_text += f"  Best: {best_loss:.4f}\n"
            summary_text += f"  Recall: {recall_str}\n"
            
            # 시간 정보 추가
            if 'time_statistics' in train_history:
                stats = train_history['time_statistics']
                summary_text += f"  Avg: {stats['average_epoch_time']:.2f}s/epoch\n"
            elif 'epoch_times' in train_history and len(train_history['epoch_times']) > 0:
                epoch_times = train_history['epoch_times']
                avg_time = sum(epoch_times) / len(epoch_times)
                summary_text += f"  Avg: {avg_time:.2f}s/epoch\n"
            
            # 총 학습 시간 계산 (여러 방법 시도)
            total_time_shown = False
            if 'total_training_time_formatted' in train_history:
                summary_text += f"  Total: {train_history['total_training_time_formatted']}\n"
                total_time_shown = True
            elif 'cumulative_time' in train_history and len(train_history['cumulative_time']) > 0:
                total_seconds = train_history['cumulative_time'][-1]
                if total_seconds > 0:
                    total_time_str = str(timedelta(seconds=int(total_seconds)))
                    summary_text += f"  Total: {total_time_str}\n"
                    total_time_shown = True
            elif 'epoch_times' in train_history and len(train_history['epoch_times']) > 0:
                # epoch_times로부터 총 시간 계산
                total_seconds = sum(train_history['epoch_times'])
                if total_seconds > 0:
                    total_time_str = str(timedelta(seconds=int(total_seconds)))
                    summary_text += f"  Total: {total_time_str}\n"
                    total_time_shown = True
            
            if not total_time_shown:
                summary_text += f"  Total: N/A\n"
            
            summary_text += "\n"
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        # Recall history 그래프 설정 (있는 경우)
        if has_recall_history:
            # 이미 axes[1, 1]에서 recall history를 그렸으므로 추가 설정만
            pass
    else:
        # Recall 정보가 없는 경우 빈 축들 숨기기
        if has_recall or has_recall_history:
            # 2x3 레이아웃인 경우
            pass
        else:
            # 2x2 레이아웃인 경우는 이미 처리됨
            pass
    
    plt.tight_layout()
    
    # 저장
    save_path = "experiment_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    
    plt.show()
    
    # 성능 요약 출력 (시간 정보 포함)
    print("\n" + "="*80)
    print("PERFORMANCE & TIMING SUMMARY")
    print("="*80)
    for data in experiment_data:
        exp_id, config, train_history = data[:3]
        recall = data[3] if len(data) > 3 else None
        recall_hist = data[4] if len(data) > 4 else None
        
        loss_func = config.get('loss_function', 'Unknown')
        final_total = train_history["total_loss"][-1]
        best_total = min(train_history["total_loss"])
        best_epoch = np.argmin(train_history["total_loss"]) + 1
        
        print(f"{exp_id[:8]}... ({loss_func}):")
        print(f"  Final Loss: {final_total:.4f}")
        print(f"  Best Loss: {best_total:.4f} (Epoch {best_epoch})")
        
        if recall_hist is not None and len(recall_hist[1]) > 0:
            final_recall = recall_hist[1][-1]
            print(f"  Final Recall@{config.get('k', 5)}: {final_recall:.4f}")
        elif recall is not None:
            print(f"  Recall@{config.get('k', 5)}: {recall:.4f}")
        
        # 시간 정보 추가
        if 'time_statistics' in train_history:
            stats = train_history['time_statistics']
            print(f"  Average Time per Epoch: {stats['average_epoch_time']:.2f}s")
            print(f"  Time Range: {stats['min_epoch_time']:.2f}s - {stats['max_epoch_time']:.2f}s")
        elif 'epoch_times' in train_history and len(train_history['epoch_times']) > 0:
            epoch_times = train_history['epoch_times']
            avg_time = sum(epoch_times) / len(epoch_times)
            min_time = min(epoch_times)
            max_time = max(epoch_times)
            print(f"  Average Time per Epoch: {avg_time:.2f}s")
            print(f"  Time Range: {min_time:.2f}s - {max_time:.2f}s")
        
        # 총 학습 시간 계산 (여러 방법 시도)
        total_time_shown = False
        if 'total_training_time_formatted' in train_history:
            print(f"  Total Training Time: {train_history['total_training_time_formatted']}")
            total_time_shown = True
        elif 'cumulative_time' in train_history and len(train_history['cumulative_time']) > 0:
            total_seconds = train_history['cumulative_time'][-1]
            if total_seconds > 0:
                total_time_str = str(timedelta(seconds=int(total_seconds)))
                print(f"  Total Training Time: {total_time_str}")
                total_time_shown = True
        elif 'epoch_times' in train_history and len(train_history['epoch_times']) > 0:
            # epoch_times로부터 총 시간 계산
            total_seconds = sum(train_history['epoch_times'])
            if total_seconds > 0:
                total_time_str = str(timedelta(seconds=int(total_seconds)))
                print(f"  Total Training Time: {total_time_str}")
                total_time_shown = True
        
        if not total_time_shown:
            print(f"  Total Training Time: N/A")
        
        print()

if __name__ == "__main__":
    main()
