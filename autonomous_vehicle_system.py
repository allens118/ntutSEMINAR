import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Set, Dict
import random
import math

class Vehicle:
    def __init__(self, vid: int, position: tuple, velocity: tuple):
        """
        初始化車輛物件 / Initialize vehicle object
        
        Args:
            vid: 車輛ID / Vehicle ID
            position: 初始位置 / Initial position
            velocity: 初始速度 / Initial velocity
        """
        self.vid = vid
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.radius = random.uniform(5, 10)  # 通信半徑 / Communication radius
        self.neighbors: Set[Vehicle] = set()  # 鄰居集合 / Set of neighbors
        self.f_value = 0  # 主要車輛標記 / Primary vehicle marker
        self.delta = 0    # 車輛評分值 / Vehicle score value
        self.cluster = None  # 所屬集群 / Belonging cluster
        self.join_requests: List[Vehicle] = []
        
        # 車輛性能指標 / Vehicle performance indicators
        self.perception_range = random.uniform(30, 50)  # 感知範圍 / Perception range
        self.processing_power = random.uniform(0.5, 1.0)  # 處理能力 / Processing power
        self.communication_quality = random.uniform(0.7, 1.0)  # 通信質量 / Communication quality
        
    def broadcast_hello(self) -> None:
        """廣播 Hello 封包"""
        pass  # 在實際系統中實現通信
        
    def calculate_delta(self, vehicles: List['Vehicle']) -> float:
        """
        計算車輛的 delta 值 / Calculate vehicle's delta value
        
        考慮因素 / Considering factors:
        - 鄰居數量 / Number of neighbors
        - 覆蓋面積 / Coverage area
        - 處理效率 / Processing efficiency
        """
        neighbor_count = len(self.neighbors)
        coverage_area = math.pi * (self.radius ** 2)
        processing_efficiency = self.processing_power
        
        # 綜合評分 / Comprehensive scoring
        self.delta = (
            neighbor_count * 0.4 +  # 鄰居權重 / Neighbor weight
            coverage_area * 0.3 +   # 覆蓋權重 / Coverage weight
            processing_efficiency * 0.3  # 效率權重 / Efficiency weight
        )
        return self.delta
        
    def update_neighbors(self, vehicles: List['Vehicle']) -> None:
        """更新鄰居列表"""
        self.neighbors.clear()
        for v in vehicles:
            if v != self and np.linalg.norm(self.position - v.position) <= self.radius:
                self.neighbors.add(v)

    def calculate_angle(self, other: 'Vehicle') -> float:
        """計算與其他車輛的夾角"""
        v1 = self.velocity
        v2 = other.velocity
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.arccos(dot_product / norms) if norms > 0 else 0

class VehicleSystem:
    def __init__(self, num_vehicles: int, map_size: tuple):
        """
        初始化車輛系統 / Initialize vehicle system
        
        Args:
            num_vehicles: 車輛數量 / Number of vehicles
            map_size: 地圖大小 / Map size
        """
        self.num_vehicles = num_vehicles
        self.map_size = map_size
        self.vehicles = self._initialize_vehicles()
        self.clusters: List[Set[Vehicle]] = []
        
    def _initialize_vehicles(self) -> List[Vehicle]:
        """
        初始化車輛位置和屬性 / Initialize vehicle positions and attributes
        
        Returns:
            車輛列表 / List of vehicles
        """
        vehicles = []
        for i in range(self.num_vehicles):
            pos = (random.uniform(0, self.map_size[0]), 
                  random.uniform(0, self.map_size[1]))
            vel = (random.uniform(-2, 2), random.uniform(-2, 2))
            vehicles.append(Vehicle(i, pos, vel))
        return vehicles

    def algorithm1_primary_selection(self) -> List[Vehicle]:
        """
        演算法 1：主要車輛選擇 / Algorithm 1: Primary Vehicle Selection
        
        選擇具有最高 delta 值的車輛作為主要車輛
        Select vehicles with highest delta values as primary vehicles
        """
        # 初始化
        for v in self.vehicles:
            v.f_value = 0
            v.broadcast_hello()
            v.update_neighbors(self.vehicles)
        
        # 計算每個車輛的 Δᵢ 值
        for v in self.vehicles:
            v.delta = v.calculate_delta(self.vehicles)
        
        # 調整目標主要車輛數量
        target_primary = max(int(len(self.vehicles) * 0.25), 5)  # 增加到25%
        
        # 根據 delta 值排序
        sorted_vehicles = sorted(self.vehicles, key=lambda v: v.delta, reverse=True)
        
        # 選擇前 N 個車輛作為主要車輛
        for v in sorted_vehicles[:target_primary]:
            v.f_value = 1
        
        return [v for v in self.vehicles if v.f_value > 0]

    def algorithm2_pareto_optimal(self, population_size: int = 50) -> np.ndarray:
        """
        演算法 2：Pareto 最優解 / Algorithm 2: Pareto Optimal Solution
        
        使用多目標最佳化尋找 Pareto 最優解
        Use multi-objective optimization to find Pareto optimal solutions
        
        優化目標 / Optimization objectives:
        - 覆蓋率(Q) / Coverage rate
        - 效率(E) / Efficiency
        - 可靠性(R) / Reliability
        """
        def initialize_population(size: int) -> np.ndarray:
            # 確保至少有一個可行解
            population = np.random.randint(2, size=(size-1, len(self.vehicles)))
            # 添加一個全1解
            population = np.vstack([population, np.ones(len(self.vehicles))])
            return population
            
        def calculate_objectives(x: np.ndarray) -> tuple:
            # 計算 Qᵢ(x), Eᵢ(x), Rᵢ(x)
            selected_vehicles = [v for i, v in enumerate(self.vehicles) if x[i]]
            if not selected_vehicles:
                return 0, 0, 0
                
            Q = len(selected_vehicles) / len(self.vehicles)  # 覆蓋率
            E = sum(v.processing_power for v in selected_vehicles) / len(selected_vehicles)  # 效率
            R = sum(v.communication_quality for v in selected_vehicles) / len(selected_vehicles)  # 可靠性
            return Q, E, R
            
        def non_dominated_sort(population: np.ndarray) -> List[np.ndarray]:
            """非支配排序"""
            n = len(population)
            domination_count = np.zeros(n)
            dominated_solutions = [[] for _ in range(n)]
            fronts = [[]]
            
            for i in range(n):
                obj_i = calculate_objectives(population[i])
                for j in range(n):
                    if i != j:
                        obj_j = calculate_objectives(population[j])
                        # i 支配 j
                        if all(o1 >= o2 for o1, o2 in zip(obj_i, obj_j)) and any(o1 > o2 for o1, o2 in zip(obj_i, obj_j)):
                            dominated_solutions[i].append(j)
                        # j 支配 i
                        elif all(o1 <= o2 for o1, o2 in zip(obj_i, obj_j)) and any(o1 < o2 for o1, o2 in zip(obj_i, obj_j)):
                            domination_count[i] += 1
                            
                if domination_count[i] == 0:
                    fronts[0].append(i)
                    
            current_front = 0
            while fronts[current_front]:
                next_front = []
                for i in fronts[current_front]:
                    for j in dominated_solutions[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front += 1
                fronts.append(next_front)
                
            return [population[front] for front in fronts[:-1] if front]
            
        def calculate_crowding_distance(front: np.ndarray) -> np.ndarray:
            """計算擁擠度距離"""
            if len(front) <= 2:
                return np.ones(len(front)) * np.inf
                
            distances = np.zeros(len(front))
            objectives = np.array([calculate_objectives(x) for x in front])
            
            for obj_index in range(3):  # 3 個目標函數
                sorted_indices = np.argsort(objectives[:, obj_index])
                distances[sorted_indices[0]] = np.inf
                distances[sorted_indices[-1]] = np.inf
                
                obj_range = objectives[sorted_indices[-1], obj_index] - objectives[sorted_indices[0], obj_index]
                if obj_range == 0:
                    continue
                    
                for i in range(1, len(front) - 1):
                    distances[sorted_indices[i]] += (
                        objectives[sorted_indices[i + 1], obj_index] -
                        objectives[sorted_indices[i - 1], obj_index]
                    ) / obj_range
                    
            return distances
        
        # 主要演算法流程
        P = initialize_population(population_size)
        if len(P) == 0:
            return np.zeros(len(self.vehicles))
            
        generations = 10  # 減少迭代次數
        
        for gen in range(generations):
            offspring = self._crossover_and_mutation(P)
            combined = np.vstack((P, offspring))
            fronts = non_dominated_sort(combined)
            
            if not fronts:
                return np.ones(len(self.vehicles))  # 返回一個預設解
                
            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) <= population_size:
                    new_pop.extend(front)
                else:
                    remaining = population_size - len(new_pop)
                    new_pop.extend(front[:remaining])
                    break
            
            if not new_pop:
                return np.ones(len(self.vehicles))
                
            P = np.array(new_pop)
        
        # 簡化最優解選擇
        return P[0]

    def _crossover_and_mutation(self, population: np.ndarray) -> np.ndarray:
        """
        執行交叉和變異操作 / Perform crossover and mutation operations
        
        用於基因演算法優化 / Used for genetic algorithm optimization
        """
        offspring = population.copy()
        n_offspring = len(offspring)
        
        # 單點交叉
        for i in range(0, n_offspring - 1, 2):
            if random.random() < 0.8:  # 交叉概率
                point = random.randint(1, len(offspring[i]) - 1)
                offspring[i, point:], offspring[i + 1, point:] = (
                    offspring[i + 1, point:].copy(),
                    offspring[i, point:].copy()
                )
        
        # 變異
        mutation_rate = 0.1
        for i in range(n_offspring):
            for j in range(len(offspring[i])):
                if random.random() < mutation_rate:
                    offspring[i, j] = 1 - offspring[i, j]  # 翻轉比特
                    
        return offspring

    def algorithm3_cluster_formation(self, beta: float = 0.3) -> List[Set[Vehicle]]:
        """
        演算法 3：集群形成 / Algorithm 3: Cluster Formation
        
        基於主要車輛形成集群 / Form clusters based on primary vehicles
        使用動態距離閾值 / Use dynamic distance threshold
        
        Args:
            beta: 角度閾值 / Angle threshold
        """
        self.clusters = []
        
        # 修改距離閾值計算
        def get_distance_threshold(v1: Vehicle, v2: Vehicle) -> float:
            base_distance = v1.radius + v2.radius
            # 增加基礎距離閾值
            return base_distance * 4  # 增加距離閾值
        
        # 重置所有車輛的集群狀態
        for v in self.vehicles:
            v.cluster = None
        
        # 對每個主要車輛形成集群
        for primary in [v for v in self.vehicles if v.f_value == 1]:
            new_cluster = {primary}
            primary.cluster = new_cluster
            
            # 尋找附近的車輛
            for other in self.vehicles:
                if other != primary and not other.cluster:
                    distance = np.linalg.norm(primary.position - other.position)
                    threshold = get_distance_threshold(primary, other)
                    
                    # 簡化集群形成條件
                    if distance <= threshold:
                        new_cluster.add(other)
                        other.cluster = new_cluster
            
            # 只添加大於最小規模的集群
            if len(new_cluster) >= 2:  # 至少包含主要車輛和一個其他車輛
                self.clusters.append(new_cluster)
        
        return self.clusters

    def visualize(self, ax):
        """視覺化當前狀態"""
        ax.clear()
        ax.set_xlim(0, self.map_size[0])
        ax.set_ylim(0, self.map_size[1])
        
        # 繪製所有車輛
        positions = np.array([v.position for v in self.vehicles])
        ax.scatter(positions[:, 0], positions[:, 1], c='blue', label='Vehicles')
        
        # 繪製主要車輛
        primary = [v for v in self.vehicles if v.f_value > 0]
        if primary:
            primary_pos = np.array([v.position for v in primary])
            ax.scatter(primary_pos[:, 0], primary_pos[:, 1], 
                      c='red', s=100, label='Primary')
        
        # 繪製集群
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.clusters)))
        for cluster, color in zip(self.clusters, colors):
            for v1 in cluster:
                for v2 in cluster:
                    if v1 != v2:
                        ax.plot([v1.position[0], v2.position[0]],
                               [v1.position[1], v2.position[1]],
                               c=color, alpha=0.3)
        
        ax.grid(True)
        ax.legend() 