# 自主車輛集群形成系統 / Autonomous Vehicle Clustering System

本專案實現了一個自主車輛集群形成的模擬系統，包含三個主要演算法。
This project implements a simulation system for autonomous vehicle cluster formation, including three main algorithms.

## 系統架構 / System Architecture

### 1. 主要車輛選擇 / Primary Vehicle Selection
- 基於車輛的 delta 值選擇主要車輛 / Select primary vehicles based on delta values
- 考慮因素 / Considering factors:
  - 鄰居數量 / Number of neighbors
  - 覆蓋面積 / Coverage area
  - 處理效率 / Processing efficiency

### 2. Pareto 最優解 / Pareto Optimal Solution
- 多目標最佳化 / Multi-objective optimization
- 優化目標 / Optimization objectives:
  - 覆蓋率 (Q) / Coverage rate
  - 效率 (E) / Efficiency
  - 可靠性 (R) / Reliability

### 3. 集群形成 / Cluster Formation
- 基於主要車輛形成集群 / Form clusters based on primary vehicles
- 使用動態距離閾值 / Use dynamic distance threshold
- 考慮車輛性能特徵 / Consider vehicle performance characteristics

## 模擬結果 / Simulation Results

### 演算法執行流程 / Algorithm Execution Flow
![Simulation Results](https://github.com/allens118/ntutSEMINAR/raw/main/docs/images/simulation_results.png)

模擬結果展示了系統的四個主要階段：
The simulation results show four main stages of the system:

1. **初始狀態 / Initial State** (左上 / Top Left)
   - 顯示所有車輛的初始分布
   - Shows initial distribution of all vehicles

2. **主要車輛選擇 / Primary Vehicle Selection** (右上 / Top Right)
   - 選出 12 輛主要車輛（紅色）
   - Selected 12 primary vehicles (in red)

3. **Pareto 最優化 / Pareto Optimization** (左下 / Bottom Left)
   - 基於多目標優化選擇車輛
   - Vehicle selection based on multi-objective optimization

4. **集群形成 / Cluster Formation** (右下 / Bottom Right)
   - 形成 9 個集群
   - Formed 9 clusters
   - 平均集群大小：2.3 輛車
   - Average cluster size: 2.3 vehicles
   - 覆蓋率：44.0%
   - Coverage rate: 44.0%

### 系統狀態 / System Status
- 總車輛數：50 / Total Vehicles: 50
- 主要車輛：12 (24.0%) / Primary Vehicles: 12 (24.0%)
- Pareto 選中：50 (100.0%) / Pareto Selected: 50 (100.0%)
- 集群數量：9 / Number of Clusters: 9
- 平均集群大小：2.3 / Average Cluster Size: 2.3
- 覆蓋率：44.0% / Coverage Rate: 44.0%

## 檔案結構 / File Structure 