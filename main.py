import matplotlib.pyplot as plt
import matplotlib.animation as animation
from autonomous_vehicle_system import VehicleSystem
import time
import numpy as np

def main():
    # 初始化系統
    num_vehicles = 50  # 增加車輛數量
    map_size = (800, 800)  # 調整地圖大小
    system = VehicleSystem(num_vehicles, map_size)
    
    # 調整圖表大小和布局
    fig = plt.figure(figsize=(15, 12))
    
    # 創建 2x2 的子圖布局，並在上方添加一個用於狀態顯示的子圖
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.3, 1])  # 添加中間行用於狀態顯示
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    status_ax = fig.add_subplot(gs[1, :])  # 跨越兩列的狀態顯示區
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    
    # 隱藏狀態顯示區的座標軸
    status_ax.axis('off')
    
    def update(frame):
        start_time = time.time()
        
        # 清空所有子圖
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
            ax.set_xlim(0, map_size[0])
            ax.set_ylim(0, map_size[1])
            ax.grid(True)
        
        status_ax.clear()
        status_ax.axis('off')
        
        try:
            # 運行三個演算法
            print(f"Frame {frame}: Running algorithms...")
            
            # Algorithm 1: Primary Vehicle Selection
            primary = system.algorithm1_primary_selection()
            print(f"Primary vehicles selected: {len(primary)}")
            
            # Algorithm 2: Pareto Optimal Solution
            pareto = system.algorithm2_pareto_optimal()
            print(f"Pareto solution found")
            
            # Algorithm 3: Cluster Formation
            clusters = system.algorithm3_cluster_formation()
            print(f"Clusters formed: {len(clusters)}")
            
            # 繪製車輛
            positions = np.array([v.position for v in system.vehicles])
            ax1.set_title('Algorithm 1: Primary Vehicle Selection\n(Initial State)', 
                         fontsize=10, pad=10)
            ax1.scatter(positions[:, 0], positions[:, 1], c='blue', label='Normal Vehicles')
            ax1.legend()
            
            # 2. 主要車輛
            ax2.set_title('Algorithm 1: Primary Vehicle Selection\n(Selected Primary Vehicles)', 
                         fontsize=10, pad=10)
            if primary:
                primary_pos = np.array([v.position for v in primary])
                ax2.scatter(primary_pos[:, 0], primary_pos[:, 1], 
                          c='red', s=100, label='Primary Vehicles')
                # 添加說明文字
                ax2.text(0.02, 0.98, 
                        f"Selected {len(primary)} primary vehicles\nbased on delta values",
                        transform=ax2.transAxes,
                        verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.8))
            ax2.legend()
            
            # 3. Pareto 最優解
            ax3.set_title('Algorithm 2: Pareto Optimal Solution\n(Multi-objective Optimization)', 
                         fontsize=10, pad=10)
            selected = [v for i, v in enumerate(system.vehicles) if pareto[i]]
            if selected:
                selected_pos = np.array([v.position for v in selected])
                ax3.scatter(selected_pos[:, 0], selected_pos[:, 1], 
                          c='green', s=80, label='Pareto Optimal')
                # 添加說明文字
                ax3.text(0.02, 0.98,
                        "Optimizing for:\n- Coverage (Q)\n- Efficiency (E)\n- Reliability (R)",
                        transform=ax3.transAxes,
                        verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.8))
            ax3.legend()
            
            # 4. 集群形成（整合視圖）
            ax4.set_title('Algorithm 3: Cluster Formation\n(Final Integration)', 
                         fontsize=10, pad=10)
            
            # 繪製所有車輛（使用更明顯的顏色和大小）
            ax4.scatter(positions[:, 0], positions[:, 1], 
                      c='lightgray', s=50, alpha=0.3, label='All Vehicles',
                      marker='o', edgecolors='gray')
            
            # 繪製主要車輛
            if primary:
                primary_pos = np.array([v.position for v in primary])
                ax4.scatter(primary_pos[:, 0], primary_pos[:, 1], 
                          c='red', s=120, alpha=0.7, label='Primary Vehicles',
                          marker='*', edgecolors='darkred')
            
            # 繪製 Pareto 最優解選中的車輛
            if selected:
                selected_pos = np.array([v.position for v in selected])
                ax4.scatter(selected_pos[:, 0], selected_pos[:, 1], 
                          c='green', s=100, alpha=0.7, label='Pareto Optimal',
                          marker='^', edgecolors='darkgreen')
            
            # 調整第四個圖的圖例
            handles, labels = ax4.get_legend_handles_labels()
            ax4.legend(handles, labels,
                      loc='center left',
                      bbox_to_anchor=(1.02, 0.5),
                      borderaxespad=0,
                      frameon=True,
                      fancybox=True,
                      shadow=True)
            
            # 繪製集群連線和集群內車輛
            cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            for i, cluster in enumerate(clusters):
                color = cluster_colors[i % len(cluster_colors)]
                
                # 計算集群的邊界框
                cluster_pos = np.array([v.position for v in cluster])
                if len(cluster_pos) > 0:
                    min_x = np.min(cluster_pos[:, 0]) - 20
                    max_x = np.max(cluster_pos[:, 0]) + 20
                    min_y = np.min(cluster_pos[:, 1]) - 20
                    max_y = np.max(cluster_pos[:, 1]) + 20
                    
                    # 繪製集群邊界
                    rect = plt.Rectangle(
                        (min_x, min_y),
                        max_x - min_x,
                        max_y - min_y,
                        fill=False,
                        linestyle='--',
                        linewidth=2,
                        edgecolor=color,
                        alpha=0.6
                    )
                    ax4.add_patch(rect)
                    
                    # 添加集群標籤
                    ax4.text(
                        min_x + (max_x - min_x)/2,
                        max_y + 10,
                        f'Cluster {i+1}',
                        horizontalalignment='center',
                        color=color,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor=color)
                    )
                
                # 繪製集群內連線（減少透明度）
                for v1 in cluster:
                    for v2 in cluster:
                        if v1 != v2:
                            ax4.plot([v1.position[0], v2.position[0]],
                                   [v1.position[1], v2.position[1]],
                                   c=color, alpha=0.2, linestyle=':',
                                   linewidth=0.5)
                
                # 繪製集群成員
                ax4.scatter(cluster_pos[:, 0], cluster_pos[:, 1], 
                          c=color, s=80, alpha=0.6, 
                          label=f'Cluster {i+1}',
                          marker='o', edgecolors='white')
            
            # 添加演算法說明
            algorithm_info = (
                "Algorithm Flow:\n"
                "1. Primary Selection: Select vehicles with highest delta values\n"
                "2. Pareto Optimization: Multi-objective optimization\n"
                "3. Cluster Formation: Group vehicles based on distance and performance"
            )
            
            # 在第四個圖的左上角添加演算法流程說明
            ax4.text(0.02, 0.98, algorithm_info,
                    transform=ax4.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5',
                             facecolor='white',
                             edgecolor='gray',
                             alpha=0.8))
            
            # 為第四個圖添加網格和更多細節
            ax4.grid(True, linestyle=':', alpha=0.6)
            ax4.set_xlabel('X Position')
            ax4.set_ylabel('Y Position')
            
            # 在中間區域顯示系統狀態
            stats_text = (
                f"System Status:\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"Total Vehicles: {len(system.vehicles)}\n"
                f"Primary Vehicles: {len(primary)} ({len(primary)/len(system.vehicles)*100:.1f}%)\n"
                f"Pareto Selected: {len(selected)} ({len(selected)/len(system.vehicles)*100:.1f}%)\n"
                f"Number of Clusters: {len(clusters)}\n"
                f"Average Cluster Size: {sum(len(c) for c in clusters)/len(clusters) if clusters else 0:.1f}\n"
                f"Coverage Rate: {len([v for v in system.vehicles if v.cluster is not None])/len(system.vehicles)*100:.1f}%"
            )
            
            # 在中間區域顯示系統狀態
            status_ax.text(0.5, 0.5, stats_text,
                         transform=status_ax.transAxes,
                         horizontalalignment='center',
                         verticalalignment='center',
                         bbox=dict(boxstyle='round,pad=0.5',
                                 facecolor='white',
                                 edgecolor='gray',
                                 alpha=0.8))
            
            print(f"Frame {frame} completed in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            print(f"Error in frame {frame}: {str(e)}")
            
        return []
    
    # 修改動畫參數
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=None,
        interval=500,     # 增加更新間隔
        blit=False,       # 關閉 blit 以避免某些情況下的問題
        cache_frame_data=False
    )
    
    # 調整圖表顯示設定
    plt.rcParams['animation.html'] = 'html5'
    
    # 關閉自動緊湊布局，改用固定間距
    plt.subplots_adjust(
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.3,
        hspace=0.4
    )
    
    # 顯示動畫
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n停止動畫...")

if __name__ == "__main__":
    main() 