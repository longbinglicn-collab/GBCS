'''
Python 3.9
'''
'''
    -----------------------
    |设置USV，UAV，TARGET数量|
    -----------------------
'''

UAV_COUNT = 20      # range(10,101,10)
USV_COUNT = 20      # range(10,101,10)
TARGET_COUNT = 10   # range(5,51,5)

import os

os.environ['OMP_NUM_THREADS'] = '1'
import matplotlib

# 桌面环境优先使用 TkAgg；服务器/无显示环境自动回退为 Agg。
_requested_backend = os.environ.get('GBCS_MPL_BACKEND')
if _requested_backend:
    matplotlib.use(_requested_backend)
else:
    try:
        matplotlib.use('TkAgg' if (os.name == 'nt' or os.environ.get('DISPLAY')) else 'Agg')
    except ImportError:
        matplotlib.use('Agg', force=True)

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
import random

# 设置随机种子（任意整数，如 42）
SEED = 123
random.seed(SEED)
np.random.seed(SEED)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 全局参数配置 (升级 + SFLA/CS参数) ====================
ARENA_SIZE_XY = 3000
ARENA_SIZE_Z = ARENA_SIZE_XY
OBSTACLES = []

# ---智能体初始化方式控制 ---
INIT_RANDOMSTATE = 0  # 1: 随机分布, 0: 均匀分布
DIS_FIG = 0 # 是否显示过程图
# --- 智能体与目标参数 ---
CAPTURE_RADIUS = 100



TARGET_IS_STATIC = 0
TARGET_RUN_NUM = 300
# MIN_CAPTURE_AGENTS = int(np.sqrt(UAV_COUNT + USV_COUNT)-1)
MIN_CAPTURE_AGENTS = (UAV_COUNT + USV_COUNT) // TARGET_COUNT -1

# --- 围捕判定模式 ---
# 'radius'：快速接近式捕获，仅要求捕获半径内人数达到阈值；
# 'geometric'：真正几何围捕，同时要求最大角度空缺不超过阈值。
CAPTURE_MODE = 'radius'
MAX_CAPTURE_ANGULAR_GAP = np.pi
# 目标预测与 SFLA 模块解耦；四个消融算法共享同一预测设置。
USE_TARGET_PREDICTION = False
# 远距离直接追击目标参考位置，进入该距离后启用共享基础几何围捕控制。
SFLA_ACTIVATION_DISTANCE = 2.0 * CAPTURE_RADIUS
# 所有 GB 消融算法共享的基础几何围捕控制器。
# 远距离直接追击；近距离使用固定半径、确定性相位的环形槽位。
USE_BASE_ENCIRCLEMENT_CONTROLLER = True
BASE_RING_RADIUS_RATIO = 0.65
BASE_RING_UAV_HEIGHT_RATIO = 0.45
# 在容量可行时，每个目标最多接收最低需求之外的冗余成员数。
MAX_EXTRA_AGENTS_PER_TARGET = 2
# CS 仅在初始、任务不可行或连续停滞时触发，并在触发后进入冷却。
CS_STAGNATION_LIMIT = 2
CS_COOLDOWN = 3
# 第二批：限制非必要任务切换，并要求切换带来足够 ETA 收益。
MAX_ASSIGNMENT_SWITCH_RATIO = 0.15
SWITCH_MIN_ETA_GAIN = 0.25
# SFLA 激活采用滞回，避免在阈值附近反复启停。
SFLA_DEACTIVATION_DISTANCE = 1.20 * SFLA_ACTIVATION_DISTANCE
SFLA_EARLY_STOP_TOL = 1e-4
SFLA_EARLY_STOP_PATIENCE = 2
# 粒球质量过低时触发 CS 结构事件。
CS_QUALITY_TRIGGER = 0.45
print("UAV_COUNT,USV_COUNT,TARGET_COUNT,MIN_CAPTURE_AGENTS", UAV_COUNT, USV_COUNT, TARGET_COUNT, MIN_CAPTURE_AGENTS)
# --- 粒球计算参数 ---
QTH = 0.6
RADIUS_CTRL = 1.0
ALPHA = 0.4
BETA = 0.6
GAMMA = 0

# --- 速度与动力学 ---
V_MAX_UAV = ARENA_SIZE_XY // 20
V_MAX_USV = int(V_MAX_UAV * 0.7)
TARGET_SPEED = V_MAX_UAV // 2
print("V_MAX_UAV,V_MAX_USV,TARGET_SPEED",V_MAX_UAV,V_MAX_USV,TARGET_SPEED)
V_MIN = 0
OMEGA_MAX = np.pi / 4

# --- 3D专用变量 ---
USV_Z = 0
TARGET_Z = 0
UAV_Z_MIN, UAV_Z_MAX = 0, ARENA_SIZE_Z

# ==================== 新增：SFLA 与 CS 算法参数 ====================
# SFLA 参数 (用于粒球内部微调)
SFLA_MEME_COUNT = 2  # 模因组数量
SFLA_MAX_STEP = 50  # 青蛙最大跳跃步长 (对应无人机速度限制)
SFLA_ITER_LOCAL = 5  # 局部迭代次数

# CS 参数 (用于粒球间重组)
CS_PA = 0.25  # 发现概率 (控制探索强度)
CS_BETA = 1.5  # 莱维飞行指数

# ==================== 新增：算法对比框架 ====================
import math
from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """所有对比算法的基类"""

    def __init__(self, env, name="Base"):
        self.env = env
        self.name = name
        self.metrics = {
            "capture_time": [],
            "total_distance": 0,
            "recluster_count": 0,
            "reconfiguration_count": 0,
            "energy_consumption": 0,
            "target_load_balance": 0.0,
            "success_rate": 0
        }

    @abstractmethod
    # @abstractmethod
    def step(self, agents, targets):
        """返回：{agent_id: target_id}"""
        pass

    def reset_metrics(self):
        """重置性能指标"""
        self.metrics = {
            "capture_time": [],
            "total_distance": 0,
            "recluster_count": 0,
            "reconfiguration_count": 0,
            "energy_consumption": 0,
            "target_load_balance": 0.0,
            "success_rate": 0
        }

    def _active_agents(self):
        """返回未被锁定为守卫的智能体，避免守卫继续参与分组和任务分配。"""
        guarding_ids = set()
        if hasattr(self.env, "_get_guarding_agent_ids"):
            guarding_ids = set(self.env._get_guarding_agent_ids())
        elif hasattr(self.env, "guarding_agents"):
            for guard_ids in self.env.guarding_agents.values():
                guarding_ids.update(int(agent_id) for agent_id in guard_ids)

        return [
            agent for agent in self.env.agents
            if int(agent[7]) not in guarding_ids
        ]


def calculate_target_load_balance(assignments, active_target_ids, active_agent_ids=None):
    """Return the coefficient of variation of active-target assignment loads."""
    active_target_ids = list(active_target_ids)
    if not active_target_ids:
        return 0.0

    if active_agent_ids is None:
        active_agent_ids = set(assignments)
    else:
        active_agent_ids = set(active_agent_ids)

    target_to_index = {target_id: index for index, target_id in enumerate(active_target_ids)}
    loads = np.zeros(len(active_target_ids), dtype=float)
    for agent_id, target_id in assignments.items():
        if agent_id in active_agent_ids and target_id in target_to_index:
            loads[target_to_index[target_id]] += 1

    mean_load = np.mean(loads)
    if mean_load <= 1e-12:
        return 0.0
    return float(np.std(loads) / mean_load)


def count_reconfigurations(previous_assignments, current_assignments, active_agent_ids=None):
    """Count agents whose effective target assignment changed after initialization."""
    if not previous_assignments:
        return 0

    if active_agent_ids is None:
        active_agent_ids = set(current_assignments)
    else:
        active_agent_ids = set(active_agent_ids)

    return sum(
        agent_id in previous_assignments
        and previous_assignments[agent_id] != target_id
        for agent_id, target_id in current_assignments.items()
        if agent_id in active_agent_ids
    )


def calculate_success_rate(captured_target_ids, total_targets):
    """Return the fraction of targets captured in the current run."""
    if total_targets <= 0:
        return 0.0
    return len(captured_target_ids) / total_targets


# ==================== 算法1：你的GB-SFLA（基于现有代码重构） ====================
class GBSFLACSAlgorithm(BaseAlgorithm):
    """有效的粒球-SFLA-CS混合围捕算法。

    设计原则：
    1. 粒球仅保存由 ``env.agents`` 同步得到的真实状态；
    2. SFLA 优化可执行围捕航点，不修改真实坐标或粒球观测；
    3. CS 优化完整分配/结构候选，只有综合代价下降时才接受；
    4. 目标分配先满足最低围捕容量，再优化距离、负载与重配置。
    """

    class GranularBallNode:
        def __init__(self, ball_id, agent_ids=None, parent=None):
            self.ball_id = int(ball_id)
            self.agent_ids = list(agent_ids or [])
            self.parent = parent
            self.children = []
            self.children_quality = []
            self.is_leaf = True
            self.is_empty = False
            self.points = np.empty((0, 3), dtype=float)
            self.center = np.zeros(3, dtype=float)
            self.radius = 0.0
            self.density = 0.0
            self.quality = 0.0

        def sync_from_agent_map(self, agent_map):
            self.agent_ids = [
                int(agent_id) for agent_id in self.agent_ids
                if int(agent_id) in agent_map
            ]
            if not self.agent_ids:
                self.points = np.empty((0, 3), dtype=float)
                self.center = np.zeros(3, dtype=float)
                self.radius = 0.0
                self.density = 0.0
                self.is_empty = True
                return

            self.points = np.asarray(
                [agent_map[agent_id][:3] for agent_id in self.agent_ids],
                dtype=float,
            )
            self.center = np.mean(self.points, axis=0)
            if len(self.points) > 1:
                self.radius = float(np.max(np.linalg.norm(self.points - self.center, axis=1)))
            else:
                self.radius = 0.0
            volume = (4.0 / 3.0) * np.pi * max(self.radius, 1e-6) ** 3
            self.density = float(len(self.points) / (volume + 1e-12))
            self.is_empty = False

        def update_features(self):
            """兼容环境绘图/同步代码；只根据当前 points 更新几何属性。"""
            points = np.asarray(self.points, dtype=float)
            if points.size == 0:
                self.points = np.empty((0, 3), dtype=float)
                self.center = np.zeros(3, dtype=float)
                self.radius = 0.0
                self.density = 0.0
                self.is_empty = True
                return
            self.points = points.reshape((-1, 3))
            self.center = np.mean(self.points, axis=0)
            self.radius = float(np.max(np.linalg.norm(self.points - self.center, axis=1))) if len(self.points) > 1 else 0.0
            volume = (4.0 / 3.0) * np.pi * max(self.radius, 1e-6) ** 3
            self.density = float(len(self.points) / (volume + 1e-12))
            self.is_empty = False

    def __init__(self, env, name="GB-CS-SFLA", use_sfla=False, use_cs=True):
        super().__init__(env, name)
        # 四个粒球消融版本共享同一主流程，仅通过这两个开关移除模块。
        self.use_sfla = bool(use_sfla)
        self.use_cs = bool(use_cs)
        self.QTH = 0.60
        self.split_tolerance = 0.035
        self.merge_tolerance = 0.025
        self.accept_tolerance = 1e-9

        self.sfla_population_size = 8
        self.sfla_memeplex_count = 2
        self.sfla_local_iterations = 3

        self.cs_nests = 6
        self.cs_iterations = 2
        self.cs_pa = CS_PA
        self.cs_beta = CS_BETA

        self.weights = {
            "uncovered": 100.0,
            "completion": 30.0,
            "angular": 20.0,
            "makespan": 10.0,
            "distance": 4.0,
            "balance": 3.0,
            "reconfiguration": 2.0,
            "energy": 1.0,
            "group": 2.0,
        }

        self.rng = np.random.default_rng(SEED + 701)
        self.next_ball_id = 0
        self.leaf_balls = []
        self.ball_tree = None
        self.all_balls = {}
        self.desired_waypoints = {}
        self.last_assignments = {}
        self.last_objective = np.inf
        self._agent_map_cache = None
        self._target_reference_cache = {}
        self._speed_cache = {}
        # SFLA 跨时间步热启动与槽位连续性状态。
        self.previous_sfla_plans = {}
        self.previous_slot_waypoints = {}
        # 共享近距离围捕激活状态。保留 sfla_active_targets 作为兼容别名。
        self.encirclement_active_targets = set()
        self.sfla_active_targets = self.encirclement_active_targets
        self.sfla_slot_continuity_weight = 0.35
        self.sfla_early_stop_tolerance = SFLA_EARLY_STOP_TOL
        self.sfla_early_stop_patience = SFLA_EARLY_STOP_PATIENCE
        # CS 事件触发状态。
        self.cs_stagnation_steps = 0
        self.cs_cooldown_remaining = 0
        self.last_cs_priority = None
        self.last_active_target_signature = None
        self.last_cs_event = None
        self.metrics['use_sfla'] = int(self.use_sfla)
        self.metrics['use_cs'] = int(self.use_cs)
        self.metrics['sfla_waypoint_steps'] = 0
        self.metrics['cs_structure_accepts'] = 0
        self.metrics['cs_assignment_accepts'] = 0
        self.metrics['cs_trigger_count'] = 0
        self.metrics['cs_cooldown_skips'] = 0
        self.metrics['assignment_makespan'] = 0.0
        self.metrics['assignment_completion_gap'] = 0.0
        self.metrics['assignment_angular_gap'] = 0.0
        self.metrics['switch_budget_used'] = 0
        self.metrics['released_redundant_agents'] = 0
        self.metrics['cs_event_target_change'] = 0
        self.metrics['cs_event_constraint'] = 0
        self.metrics['cs_event_stagnation'] = 0
        self.metrics['cs_event_quality'] = 0
        self.metrics['sfla_warm_start_hits'] = 0
        self.metrics['sfla_early_stops'] = 0
        self.metrics['sfla_activation_count'] = 0
        self.metrics['base_encirclement_activation_count'] = 0
        self.metrics['base_encirclement_waypoint_steps'] = 0

    # ------------------------------------------------------------------
    # 基础状态和粒球几何
    # ------------------------------------------------------------------
    def _get_next_id(self):
        value = self.next_ball_id
        self.next_ball_id += 1
        return value

    def _active_target_ids(self):
        return [
            target_id for target_id in range(len(self.env.targets))
            if target_id not in self.env.permanently_captured
        ]

    def _begin_step_cache(self):
        active_agents = BaseAlgorithm._active_agents(self)
        self._agent_map_cache = {int(agent[7]): agent for agent in active_agents}
        self._target_reference_cache = {}
        self._speed_cache = {
            int(agent[7]): float(V_MAX_UAV if int(agent[6]) == 0 else V_MAX_USV)
            for agent in active_agents
        }

    def _active_agent_map(self):
        if self._agent_map_cache is not None:
            return self._agent_map_cache
        return {int(agent[7]): agent for agent in BaseAlgorithm._active_agents(self)}

    def _max_speed(self, agent_id):
        agent_id = int(agent_id)
        if agent_id in self._speed_cache:
            return self._speed_cache[agent_id]
        agent = self._active_agent_map().get(agent_id)
        if agent is None:
            for candidate in self.env.agents:
                if int(candidate[7]) == int(agent_id):
                    agent = candidate
                    break
        if agent is None:
            return float(V_MAX_UAV)
        return float(V_MAX_UAV if int(agent[6]) == 0 else V_MAX_USV)

    def _new_ball(self, agent_ids, parent=None):
        ball = self.GranularBallNode(self._get_next_id(), agent_ids, parent=parent)
        ball.sync_from_agent_map(self._active_agent_map())
        ball.quality = self._calculate_cluster_quality(ball)
        return ball

    def _clone_partition(self, balls):
        return [self._new_ball(list(ball.agent_ids)) for ball in balls if ball.agent_ids]

    def _refresh_tree(self):
        self.all_balls = {ball.ball_id: ball for ball in self.leaf_balls}
        if not self.leaf_balls:
            self.ball_tree = None
        elif len(self.leaf_balls) == 1:
            self.ball_tree = self.leaf_balls[0]
            self.ball_tree.parent = None
        else:
            root_ids = [agent_id for ball in self.leaf_balls for agent_id in ball.agent_ids]
            root = self.GranularBallNode(-1, root_ids)
            root.sync_from_agent_map(self._active_agent_map())
            root.is_leaf = False
            root.children = self.leaf_balls
            root.children_quality = [float(ball.quality) for ball in self.leaf_balls]
            for ball in self.leaf_balls:
                ball.parent = root
                ball.is_leaf = True
            self.ball_tree = root
            self.all_balls[root.ball_id] = root
        self.env.ball_tree = self.ball_tree

    def _sync_real_balls(self):
        """用真实智能体位置更新粒球成员和几何，优化器不得改写真实状态。"""
        agent_map = self._active_agent_map()
        active_ids = sorted(agent_map)
        if not active_ids:
            self.leaf_balls = []
            self._refresh_tree()
            return

        if not self.leaf_balls:
            self.leaf_balls = [self._new_ball(active_ids)]
            self._refresh_tree()
            return

        # 先用上一步成员同步中心，再按最近真实中心重新归属。
        valid_balls = []
        for ball in self.leaf_balls:
            ball.agent_ids = [agent_id for agent_id in ball.agent_ids if agent_id in agent_map]
            ball.sync_from_agent_map(agent_map)
            if ball.agent_ids:
                valid_balls.append(ball)
        self.leaf_balls = valid_balls or [self._new_ball(active_ids)]

        centers = np.asarray([ball.center for ball in self.leaf_balls], dtype=float)
        memberships = {ball.ball_id: [] for ball in self.leaf_balls}
        for agent_id in active_ids:
            position = np.asarray(agent_map[agent_id][:3], dtype=float)
            nearest_index = int(np.argmin(np.linalg.norm(centers - position, axis=1)))
            memberships[self.leaf_balls[nearest_index].ball_id].append(agent_id)

        synced = []
        for ball in self.leaf_balls:
            ball.agent_ids = memberships[ball.ball_id]
            ball.sync_from_agent_map(agent_map)
            if ball.agent_ids:
                ball.quality = self._calculate_cluster_quality(ball)
                synced.append(ball)
        self.leaf_balls = synced or [self._new_ball(active_ids)]
        self._refresh_tree()

    def _calculate_cluster_quality(self, ball):
        n = len(ball.agent_ids)
        if n == 0:
            return 0.0
        if n == 1:
            return 0.25

        points = np.asarray(ball.points, dtype=float)
        avg_distance = float(np.mean(np.linalg.norm(points - ball.center, axis=1)))
        tightness = 1.0 / (1.0 + avg_distance / (ARENA_SIZE_XY * 0.1 + 1e-12))

        ideal_min = max(1, int(MIN_CAPTURE_AGENTS))
        ideal_max = ideal_min + 3
        if n < ideal_min:
            scale = n / ideal_min
        elif n <= ideal_max:
            scale = 1.0
        else:
            scale = max(0.35, 1.0 - 0.05 * (n - ideal_max))

        agent_map = self._active_agent_map()
        uav_count = sum(int(agent_map[agent_id][6]) == 0 for agent_id in ball.agent_ids if agent_id in agent_map)
        usv_count = n - uav_count
        if uav_count == 0 or usv_count == 0:
            heterogeneity = 0.5
        else:
            heterogeneity = 4.0 * uav_count * usv_count / (n * n)

        quality = ALPHA * tightness + BETA * scale + GAMMA * heterogeneity
        return float(np.clip(quality, 0.0, 1.0))

    def _ball_loss(self, ball):
        n = len(ball.agent_ids)
        min_agents = max(1, int(MIN_CAPTURE_AGENTS))
        shortage = max(0, min_agents - n) / min_agents
        oversize = max(0, n - (min_agents + 3)) / max(min_agents, 1)
        return float((1.0 - ball.quality) + 0.35 * shortage + 0.05 * oversize)

    def _partition_loss(self, balls, active_target_count):
        valid = [ball for ball in balls if ball.agent_ids]
        if not valid:
            return float("inf")
        total_agents = sum(len(ball.agent_ids) for ball in valid)
        weighted = sum(len(ball.agent_ids) * self._ball_loss(ball) for ball in valid) / max(total_agents, 1)
        feasible_group_count = max(1, min(active_target_count, total_agents // max(MIN_CAPTURE_AGENTS, 1)))
        count_gap = abs(len(valid) - feasible_group_count) / feasible_group_count
        undersized = sum(len(ball.agent_ids) < MIN_CAPTURE_AGENTS for ball in valid) / len(valid)
        return float(weighted + 0.12 * count_gap + 0.20 * undersized)

    # ------------------------------------------------------------------
    # 粒球分裂与合并：均在真实成员 ID 分区上进行
    # ------------------------------------------------------------------
    def _split_candidate(self, ball):
        minimum_child = max(2, int(MIN_CAPTURE_AGENTS))
        if len(ball.agent_ids) < 2 * minimum_child:
            return None
        if len(np.unique(np.round(ball.points, 8), axis=0)) < 2:
            return None

        labels = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit_predict(ball.points)
        ids = np.asarray(ball.agent_ids, dtype=int)
        children_ids = [ids[labels == label].tolist() for label in (0, 1)]
        if any(len(group) < minimum_child for group in children_ids):
            return None
        return [self._new_ball(group) for group in children_ids]

    def _try_deterministic_split(self, ball, active_target_count):
        children = self._split_candidate(ball)
        if children is None:
            return None
        parent_loss = self._ball_loss(ball)
        child_loss = sum(len(child.agent_ids) * self._ball_loss(child) for child in children) / len(ball.agent_ids)
        capacity_pressure = len(self.leaf_balls) < active_target_count
        if child_loss + self.split_tolerance < parent_loss:
            return children
        if capacity_pressure and child_loss <= parent_loss + self.split_tolerance:
            return children
        return None

    def _try_merge_pair(self, balls, active_target_count):
        if len(balls) <= 1:
            return None
        candidates = []
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                a, b = balls[i], balls[j]
                distance = float(np.linalg.norm(a.center - b.center))
                candidates.append((distance, i, j))
        candidates.sort()

        current_loss = self._partition_loss(balls, active_target_count)
        for _, i, j in candidates:
            a, b = balls[i], balls[j]
            if len(a.agent_ids) >= MIN_CAPTURE_AGENTS and len(b.agent_ids) >= MIN_CAPTURE_AGENTS:
                continue
            merged = self._new_ball(a.agent_ids + b.agent_ids)
            proposal = [ball for index, ball in enumerate(balls) if index not in (i, j)] + [merged]
            proposal_loss = self._partition_loss(proposal, active_target_count)
            if proposal_loss + self.merge_tolerance < current_loss:
                return proposal
            if len(balls) > active_target_count and proposal_loss <= current_loss + self.merge_tolerance:
                return proposal
        return None

    def _adapt_structure(self, active_target_count):
        operations = 0
        max_operations = max(1, active_target_count)
        while operations < max_operations:
            candidates = sorted(
                self.leaf_balls,
                key=lambda ball: (ball.quality, -len(ball.agent_ids)),
            )
            accepted = False
            for ball in candidates:
                if ball.quality >= self.QTH and len(self.leaf_balls) >= active_target_count:
                    continue
                children = self._try_deterministic_split(ball, active_target_count)
                if children is None:
                    continue
                self.leaf_balls = [candidate for candidate in self.leaf_balls if candidate is not ball] + children
                self.metrics["recluster_count"] += 1
                operations += 1
                accepted = True
                break
            if not accepted or len(self.leaf_balls) >= active_target_count:
                break

        merged = self._try_merge_pair(self.leaf_balls, active_target_count)
        if merged is not None:
            self.leaf_balls = merged
            self.metrics["recluster_count"] += 1
        for ball in self.leaf_balls:
            ball.quality = self._calculate_cluster_quality(ball)
        self._refresh_tree()

    # ------------------------------------------------------------------
    # 目标参考位置、代价与容量优先分配
    # ------------------------------------------------------------------
    def _target_reference_position(self, target_id):
        """返回统一目标参考位置；预测开关独立于 SFLA/CS 消融开关。"""
        target_id = int(target_id)
        if target_id in self._target_reference_cache:
            return self._target_reference_cache[target_id].copy()
        target = self.env.targets[target_id]
        position = np.asarray(target[:3], dtype=float).copy()

        # 静态目标或关闭预测时，所有消融算法均使用完全相同的当前位置。
        if TARGET_IS_STATIC != 0 or not USE_TARGET_PREDICTION:
            position[2] = TARGET_Z
            self._target_reference_cache[target_id] = position.copy()
            return position

        nearby = []
        for agent in self.env.agents:
            distance_xy = float(np.linalg.norm(np.asarray(agent[:2]) - position[:2]))
            if distance_xy < TARGET_RUN_NUM:
                nearby.append(np.asarray(agent[:2], dtype=float))
        if nearby:
            mean_xy = np.mean(np.asarray(nearby), axis=0)
            direction = position[:2] - mean_xy
            norm = float(np.linalg.norm(direction))
            if norm > 1e-12:
                position[:2] += TARGET_SPEED * direction / norm
        else:
            speed = float(target[3]) if len(target) > 3 else float(TARGET_SPEED)
            theta = float(target[4]) if len(target) > 4 else 0.0
            position[:2] += speed * np.array([np.cos(theta), np.sin(theta)])
        position[2] = TARGET_Z
        self._target_reference_cache[target_id] = position.copy()
        return position

    def _arrival_cost(self, agent_id, target_id):
        agent = self._active_agent_map()[int(agent_id)]
        target_position = self._target_reference_position(int(target_id))
        return float(np.linalg.norm(np.asarray(agent[:3]) - target_position) / max(self._max_speed(agent_id), 1e-9))

    def _agent_to_ball(self, balls=None):
        mapping = {}
        for ball in (balls if balls is not None else self.leaf_balls):
            for agent_id in ball.agent_ids:
                mapping[int(agent_id)] = ball.ball_id
        return mapping

    def _capacity_requirements(self, active_ids, active_target_ids):
        requirements = {target_id: 0 for target_id in active_target_ids}
        if not active_target_ids:
            return requirements
        minimum = max(1, int(MIN_CAPTURE_AGENTS))
        if len(active_ids) >= minimum * len(active_target_ids):
            return {target_id: minimum for target_id in active_target_ids}

        base, remainder = divmod(len(active_ids), len(active_target_ids))
        for index, target_id in enumerate(active_target_ids):
            requirements[target_id] = base + (1 if index < remainder else 0)
        return requirements

    def _capacity_bounds(self, active_ids, active_target_ids):
        """返回各目标最低需求和可行冗余上限。

        当总智能体数量超过 ``最低需求 + MAX_EXTRA`` 的总容量时，按目标
        顺序均匀扩展上限，只允许不可避免的最小溢出。
        """
        requirements = self._capacity_requirements(active_ids, active_target_ids)
        upper = {
            int(target_id): int(requirements[int(target_id)] + max(0, MAX_EXTRA_AGENTS_PER_TARGET))
            for target_id in active_target_ids
        }
        shortage = max(0, len(active_ids) - sum(upper.values()))
        target_ids = [int(target_id) for target_id in active_target_ids]
        index = 0
        while shortage > 0 and target_ids:
            upper[target_ids[index % len(target_ids)]] += 1
            shortage -= 1
            index += 1
        return requirements, upper

    def _target_completion_priority(self, target_id, active_ids, requirements):
        """Return a lexicographic completion priority for one active target.

        Targets needing fewer additional in-range agents are handled first.  The
        required-member ETA breaks ties, so the allocator can finish targets
        that are already close without abandoning minimum coverage of others.
        """
        target_id = int(target_id)
        requirement = max(1, int(requirements.get(target_id, MIN_CAPTURE_AGENTS)))
        agent_map = self._active_agent_map()
        target_pos = self._target_reference_position(target_id)
        distances = []
        arrivals = []
        in_range_vectors = []
        for agent_id in active_ids:
            agent = agent_map[int(agent_id)]
            delta = np.asarray(agent[:3], dtype=float) - target_pos
            distance = float(np.linalg.norm(delta))
            distances.append(distance)
            arrivals.append(self._arrival_cost(agent_id, target_id))
            if distance <= CAPTURE_RADIUS:
                in_range_vectors.append(delta[:2])

        in_range_count = len(in_range_vectors)
        remaining = max(0, requirement - in_range_count)
        arrivals.sort()
        kth_eta = arrivals[min(requirement - 1, len(arrivals) - 1)] if arrivals else float('inf')

        angular_deficit = 1.0
        if CAPTURE_MODE == 'radius':
            angular_deficit = 0.0
        elif in_range_count >= requirement and len(in_range_vectors) >= 2:
            vectors = np.asarray(in_range_vectors, dtype=float)
            valid = np.linalg.norm(vectors, axis=1) > 1e-9
            vectors = vectors[valid]
            if len(vectors) >= 2:
                angles = np.sort(np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2 * np.pi))
                max_gap = float(np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi])))
                angular_deficit = max(0.0, max_gap - MAX_CAPTURE_ANGULAR_GAP) / (2 * np.pi)
        return (float(remaining), float(angular_deficit), float(kth_eta), target_id)

    def _agent_in_capture_range(self, agent_id, target_id):
        agent = self._active_agent_map()[int(agent_id)]
        target = self._target_reference_position(int(target_id))
        return bool(np.linalg.norm(np.asarray(agent[:3], dtype=float) - target) <= CAPTURE_RADIUS)

    def _capacity_aware_assignment(self, active_target_ids, balls=None):
        active_target_ids = [int(target_id) for target_id in active_target_ids]
        agent_map = self._active_agent_map()
        active_ids = sorted(agent_map)
        if not active_ids or not active_target_ids:
            return {}

        balls = balls if balls is not None else self.leaf_balls
        agent_to_ball = self._agent_to_ball(balls)
        previous = self.last_assignments or getattr(self.env, "previous_assignments", {}) or {}
        requirements, upper_bounds = self._capacity_bounds(active_ids, active_target_ids)

        assignments = {}
        loads = {target_id: 0 for target_id in active_target_ids}
        unassigned = set(active_ids)
        ball_targets = {}

        # Completion-first order: finish targets already close to capture, while
        # still assigning every active target its minimum required team.
        target_order = sorted(
            active_target_ids,
            key=lambda target_id: self._target_completion_priority(
                target_id, active_ids, requirements
            ),
        )
        target_rank = {target_id: rank for rank, target_id in enumerate(target_order)}

        for target_id in target_order:
            for _ in range(requirements[target_id]):
                if not unassigned:
                    break
                best_agent = None
                best_cost = float("inf")
                for agent_id in unassigned:
                    cost = self._arrival_cost(agent_id, target_id)
                    if self._agent_in_capture_range(agent_id, target_id):
                        cost -= 0.75
                    if agent_id in previous:
                        cost += -0.45 if int(previous[agent_id]) == target_id else 0.45
                    ball_id = agent_to_ball.get(agent_id)
                    if ball_id in ball_targets:
                        cost += -0.25 if ball_targets[ball_id] == target_id else 0.55
                    if cost < best_cost:
                        best_cost = cost
                        best_agent = agent_id
                assignments[best_agent] = target_id
                unassigned.remove(best_agent)
                loads[target_id] += 1
                ball_id = agent_to_ball.get(best_agent)
                if ball_id is not None:
                    ball_targets.setdefault(ball_id, target_id)

        mean_requirement = max(1.0, len(active_ids) / len(active_target_ids))
        rank_scale = max(len(target_order) - 1, 1)
        for agent_id in sorted(unassigned):
            available_targets = [
                target_id for target_id in active_target_ids
                if loads[target_id] < upper_bounds[target_id]
            ]
            candidate_targets = available_targets or active_target_ids
            best_target = None
            best_cost = float("inf")
            ball_id = agent_to_ball.get(agent_id)
            for target_id in candidate_targets:
                cost = self._arrival_cost(agent_id, target_id)
                cost += 0.25 * (loads[target_id] / mean_requirement)
                cost += 0.12 * (target_rank[target_id] / rank_scale)
                if self._agent_in_capture_range(agent_id, target_id):
                    cost -= 0.50
                if agent_id in previous and int(previous[agent_id]) != target_id:
                    cost += 0.55
                if ball_id in ball_targets and ball_targets[ball_id] != target_id:
                    cost += 0.50
                if cost < best_cost:
                    best_cost = cost
                    best_target = target_id
            assignments[agent_id] = int(best_target)
            loads[int(best_target)] += 1
            if ball_id is not None:
                ball_targets.setdefault(ball_id, int(best_target))

        return self._repair_assignment_capacity(assignments, active_target_ids)

    def _repair_assignment_capacity(self, assignments, active_target_ids=None):
        active_target_ids = (
            self._active_target_ids() if active_target_ids is None
            else [int(target_id) for target_id in active_target_ids]
        )
        agent_map = self._active_agent_map()
        active_ids = sorted(agent_map)
        if not active_target_ids or not active_ids:
            return {}

        active_set = set(active_target_ids)
        repaired = {}
        for agent_id in active_ids:
            target_id = assignments.get(agent_id)
            if target_id in active_set:
                repaired[agent_id] = int(target_id)
            else:
                repaired[agent_id] = min(
                    active_target_ids,
                    key=lambda candidate: self._arrival_cost(agent_id, candidate),
                )

        requirements, upper_bounds = self._capacity_bounds(active_ids, active_target_ids)
        loads = {target_id: 0 for target_id in active_target_ids}
        for target_id in repaired.values():
            loads[target_id] += 1

        # 先修复最低围捕容量缺口。
        for deficient_target in active_target_ids:
            while loads[deficient_target] < requirements[deficient_target]:
                donors = [
                    target_id for target_id in active_target_ids
                    if loads[target_id] > requirements[target_id]
                ]
                if not donors:
                    break
                move_options = []
                for agent_id, source_target in repaired.items():
                    if source_target not in donors:
                        continue
                    increase = self._arrival_cost(agent_id, deficient_target) - self._arrival_cost(agent_id, source_target)
                    if self.last_assignments.get(agent_id) == deficient_target:
                        increase -= 0.25
                    move_options.append((increase, agent_id, source_target))
                if not move_options:
                    break
                _, agent_id, source_target = min(move_options)
                repaired[agent_id] = deficient_target
                loads[source_target] -= 1
                loads[deficient_target] += 1

        # 在总容量可行时修复过度冗余，减少几十个成员集中到同一目标。
        while True:
            overloaded = [target_id for target_id in active_target_ids if loads[target_id] > upper_bounds[target_id]]
            receivers = [target_id for target_id in active_target_ids if loads[target_id] < upper_bounds[target_id]]
            if not overloaded or not receivers:
                break
            move_options = []
            for source_target in overloaded:
                if loads[source_target] <= requirements[source_target]:
                    continue
                for agent_id, assigned_target in repaired.items():
                    if assigned_target != source_target:
                        continue
                    for destination in receivers:
                        increase = self._arrival_cost(agent_id, destination) - self._arrival_cost(agent_id, source_target)
                        if self.last_assignments.get(agent_id) == destination:
                            increase -= 0.25
                        move_options.append((increase, agent_id, source_target, destination))
            if not move_options:
                break
            _, agent_id, source_target, destination = min(move_options)
            repaired[agent_id] = destination
            loads[source_target] -= 1
            loads[destination] += 1
        return repaired

    def _stabilize_assignment(self, candidate, active_target_ids):
        """Limit nonessential target switches while preserving capacity feasibility."""
        active_target_ids = [int(target_id) for target_id in active_target_ids]
        candidate = self._repair_assignment_capacity(candidate, active_target_ids)
        active_ids = sorted(self._active_agent_map())
        previous = {
            int(agent_id): int(target_id)
            for agent_id, target_id in (self.last_assignments or {}).items()
            if int(agent_id) in active_ids and int(target_id) in active_target_ids
        }
        if not previous:
            self.metrics['switch_budget_used'] = 0
            return candidate

        stabilized = {
            agent_id: previous.get(agent_id, int(candidate[agent_id]))
            for agent_id in active_ids
        }
        requirements, upper_bounds = self._capacity_bounds(active_ids, active_target_ids)
        loads = {target_id: 0 for target_id in active_target_ids}
        for target_id in stabilized.values():
            loads[target_id] += 1

        changes = []
        for agent_id in active_ids:
            old_target = stabilized[agent_id]
            new_target = int(candidate[agent_id])
            if old_target == new_target:
                continue
            eta_gain = self._arrival_cost(agent_id, old_target) - self._arrival_cost(agent_id, new_target)
            completion_bonus = (
                0.35 if self._agent_in_capture_range(agent_id, new_target) else 0.0
            )
            changes.append((eta_gain + completion_bonus, agent_id, old_target, new_target))

        # Mandatory moves first: repair deficits and overloaded old targets.
        mandatory = set()
        for destination in active_target_ids:
            while loads[destination] < requirements[destination]:
                options = [
                    item for item in changes
                    if item[3] == destination
                    and item[1] not in mandatory
                    and loads[item[2]] > requirements[item[2]]
                ]
                if not options:
                    break
                chosen = max(options)
                _, agent_id, source, target = chosen
                stabilized[agent_id] = target
                loads[source] -= 1
                loads[target] += 1
                mandatory.add(agent_id)

        for source in active_target_ids:
            while loads[source] > upper_bounds[source]:
                options = [
                    item for item in changes
                    if item[2] == source
                    and item[1] not in mandatory
                    and loads[item[3]] < upper_bounds[item[3]]
                ]
                if not options:
                    break
                chosen = max(options)
                _, agent_id, old_target, new_target = chosen
                stabilized[agent_id] = new_target
                loads[old_target] -= 1
                loads[new_target] += 1
                mandatory.add(agent_id)

        switch_budget = int(np.ceil(MAX_ASSIGNMENT_SWITCH_RATIO * len(active_ids)))
        optional_used = 0
        for gain, agent_id, source, destination in sorted(changes, reverse=True):
            if agent_id in mandatory or optional_used >= switch_budget:
                continue
            if gain < SWITCH_MIN_ETA_GAIN:
                continue
            if loads[source] <= requirements[source] or loads[destination] >= upper_bounds[destination]:
                continue
            stabilized[agent_id] = destination
            loads[source] -= 1
            loads[destination] += 1
            optional_used += 1

        stabilized = self._repair_assignment_capacity(stabilized, active_target_ids)
        switches = sum(
            agent_id in previous and int(stabilized[agent_id]) != int(previous[agent_id])
            for agent_id in active_ids
        )
        self.metrics['switch_budget_used'] = int(switches)
        return stabilized

    def _assignment_components(self, assignments, balls=None):
        """Evaluate capacity, current completion, geometry and predicted finish time."""
        active_target_ids = self._active_target_ids()
        active_agent_map = self._active_agent_map()
        active_ids = sorted(active_agent_map)
        invalid = {
            "uncovered": float("inf"), "completion": float("inf"),
            "angular": float("inf"), "makespan": float("inf"),
            "distance": float("inf"), "balance": float("inf"),
            "reconfiguration": float("inf"), "energy": float("inf"),
            "group": float("inf"),
        }
        if not active_target_ids:
            return {key: 0.0 for key in invalid}
        if set(assignments) != set(active_ids):
            return invalid
        if any(target_id not in active_target_ids for target_id in assignments.values()):
            return invalid

        requirements = self._capacity_requirements(active_ids, active_target_ids)
        loads = {target_id: 0 for target_id in active_target_ids}
        arrivals_by_target = {target_id: [] for target_id in active_target_ids}
        in_range_by_target = {target_id: [] for target_id in active_target_ids}
        arrival_times = []
        energy_terms = []
        for agent_id, target_id in assignments.items():
            target_id = int(target_id)
            loads[target_id] += 1
            arrival = self._arrival_cost(agent_id, target_id)
            arrivals_by_target[target_id].append(arrival)
            arrival_times.append(arrival)
            agent = active_agent_map[agent_id]
            target = self._target_reference_position(target_id)
            delta = np.asarray(agent[:3], dtype=float) - target
            distance = float(np.linalg.norm(delta))
            if distance <= CAPTURE_RADIUS:
                in_range_by_target[target_id].append(int(agent_id))
            energy_terms.append((distance / max(ARENA_SIZE_XY, 1)) ** 2)

        uncovered = float(sum(max(0, requirements[t] - loads[t]) for t in active_target_ids))
        completion = float(sum(max(0, requirements[t] - len(in_range_by_target[t])) for t in active_target_ids))
        angular = 0.0
        if CAPTURE_MODE == 'geometric':
            for target_id in active_target_ids:
                ids = in_range_by_target[target_id]
                if len(ids) < requirements[target_id]:
                    angular += 1.0
                    continue
                vectors = np.asarray([
                    active_agent_map[agent_id][:2] - self._target_reference_position(target_id)[:2]
                    for agent_id in ids
                ], dtype=float)
                valid = np.linalg.norm(vectors, axis=1) > 1e-9
                vectors = vectors[valid]
                if len(vectors) < 2:
                    angular += 1.0
                    continue
                angles = np.sort(np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2 * np.pi))
                gap = float(np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi])))
                angular += max(0.0, gap - MAX_CAPTURE_ANGULAR_GAP) / (2 * np.pi)

        horizon = max(ARENA_SIZE_XY / max(min(V_MAX_UAV, V_MAX_USV), 1), 1e-9)
        completion_times = []
        for target_id in active_target_ids:
            requirement = max(1, int(requirements[target_id]))
            arrivals = sorted(arrivals_by_target[target_id])
            if len(arrivals) >= requirement:
                completion_times.append(arrivals[requirement - 1])
            else:
                completion_times.append(horizon + float(requirement - len(arrivals)))
        makespan = float(max(completion_times, default=0.0) / horizon)
        distance_norm = float(np.mean(arrival_times) / horizon) if arrival_times else 0.0
        load_values = np.asarray(list(loads.values()), dtype=float)
        balance = float(np.std(load_values) / (np.mean(load_values) + 1e-12))
        previous = self.last_assignments or getattr(self.env, "previous_assignments", {}) or {}
        reconfiguration = sum(
            agent_id in previous and int(previous[agent_id]) != int(target_id)
            for agent_id, target_id in assignments.items()
        ) / max(len(assignments), 1)
        energy = float(np.mean(energy_terms)) if energy_terms else 0.0
        valid_balls = [ball for ball in (balls if balls is not None else self.leaf_balls) if ball.agent_ids]
        group_loss = float(np.mean([1.0 - ball.quality for ball in valid_balls])) if valid_balls else 1.0
        return {
            "uncovered": uncovered,
            "completion": completion,
            "angular": float(angular),
            "makespan": makespan,
            "distance": distance_norm,
            "balance": balance,
            "reconfiguration": float(reconfiguration),
            "energy": energy,
            "group": group_loss,
        }

    def _assignment_priority(self, assignments, balls=None):
        """Lexicographic objective: feasibility, real completion, geometry, finish time."""
        c = self._assignment_components(assignments, balls)
        return (
            c["uncovered"],
            c["completion"],
            c["angular"],
            c["makespan"],
            c["distance"],
            c["reconfiguration"],
            c["balance"],
            c["energy"],
            c["group"],
        )

    def _is_priority_better(self, candidate_priority, incumbent_priority):
        """按字典序比较，避免次要指标改善牺牲任务可行性或 makespan。"""
        for candidate, incumbent in zip(candidate_priority, incumbent_priority):
            candidate = float(candidate)
            incumbent = float(incumbent)
            if candidate + self.accept_tolerance < incumbent:
                return True
            if incumbent + self.accept_tolerance < candidate:
                return False
        return False

    def _composite_cost(self, assignments, balls=None):
        """保留标量值用于日志；CS接受决策使用 `_assignment_priority`。"""
        components = self._assignment_components(assignments, balls)
        if not all(np.isfinite(value) for value in components.values()):
            return float("inf")
        return float(sum(self.weights[key] * components[key] for key in self.weights))

    def _accept_assignment_if_better(self, incumbent, candidate, balls=None):
        incumbent_priority = self._assignment_priority(incumbent, balls)
        candidate_priority = self._assignment_priority(candidate, balls)
        if self._is_priority_better(candidate_priority, incumbent_priority):
            return True, dict(candidate), candidate_priority
        return False, dict(incumbent), incumbent_priority

    def _should_run_cs(self, current_priority, active_target_ids=None):
        """Run CS only for explicit state events, not on every simulation step."""
        if not self.use_cs:
            return False

        explicit_target_event = active_target_ids is not None
        if active_target_ids is None:
            active_target_ids = self._active_target_ids()
        else:
            active_target_ids = [int(value) for value in active_target_ids]

        signature = tuple(sorted(active_target_ids))
        target_changed = (
            explicit_target_event
            and self.last_active_target_signature is not None
            and signature != self.last_active_target_signature
        )
        if explicit_target_event:
            self.last_active_target_signature = signature

        has_constraint_gap = bool(
            current_priority and float(current_priority[0]) > self.accept_tolerance
        )
        poor_structure = any(
            ball.agent_ids and float(ball.quality) < CS_QUALITY_TRIGGER
            for ball in self.leaf_balls
        )
        stagnated = self.cs_stagnation_steps >= CS_STAGNATION_LIMIT
        initial_step = not bool(self.last_assignments)

        # Cooldown blocks ordinary quality/stagnation/constraint events.  A real
        # active-target set change bypasses cooldown because the old assignment
        # is no longer the same task instance.
        if self.cs_cooldown_remaining > 0 and not target_changed and not initial_step:
            self.metrics['cs_cooldown_skips'] += 1
            return False

        events = []
        if target_changed:
            self.metrics['cs_event_target_change'] += 1
            events.append('target_change')
        if has_constraint_gap:
            self.metrics['cs_event_constraint'] += 1
            events.append('constraint')
        if stagnated:
            self.metrics['cs_event_stagnation'] += 1
            events.append('stagnation')
        if poor_structure:
            self.metrics['cs_event_quality'] += 1
            events.append('quality')
        if initial_step:
            events.append('initial')
        self.last_cs_event = '+'.join(events) if events else None
        return bool(initial_step or target_changed or has_constraint_gap or stagnated or poor_structure)

    def _update_cs_progress(self, current_priority):
        if self.last_cs_priority is None or self._is_priority_better(current_priority, self.last_cs_priority):
            self.cs_stagnation_steps = 0
        else:
            self.cs_stagnation_steps += 1
        self.last_cs_priority = tuple(float(value) for value in current_priority)

    def _levy_flight(self, beta=None):
        beta = float(self.cs_beta if beta is None else beta)
        sigma_u = (
            math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = self.rng.normal(0, sigma_u)
        v = self.rng.normal(0, 1)
        return float(u / (abs(v) ** (1 / beta) + 1e-12))

    def _accept_if_improved(self, incumbent, incumbent_cost, candidate, candidate_cost):
        if candidate_cost + self.accept_tolerance < incumbent_cost:
            return True, dict(candidate), float(candidate_cost)
        return False, dict(incumbent), float(incumbent_cost)

    def _mutate_assignment(self, assignment, active_target_ids, balls=None):
        candidate = dict(assignment)
        active_target_ids = [int(target_id) for target_id in active_target_ids]
        if len(active_target_ids) <= 1 or not candidate:
            return candidate

        balls = balls if balls is not None else self.leaf_balls
        levy = abs(self._levy_flight())
        mutation_fraction = min(0.40, max(1.0 / len(candidate), 0.04 * levy))
        mutation_count = max(1, int(np.ceil(mutation_fraction * len(candidate))))

        selected_ids = []
        if balls and self.rng.random() < 0.65:
            ball = balls[int(self.rng.integers(0, len(balls)))]
            selected_ids.extend(ball.agent_ids)
        remaining = [agent_id for agent_id in candidate if agent_id not in selected_ids]
        if len(selected_ids) < mutation_count and remaining:
            extra_count = min(mutation_count - len(selected_ids), len(remaining))
            selected_ids.extend(self.rng.choice(remaining, size=extra_count, replace=False).tolist())

        for agent_id in selected_ids[:max(mutation_count, len(selected_ids))]:
            current_target = candidate[agent_id]
            alternatives = [target_id for target_id in active_target_ids if target_id != current_target]
            if not alternatives:
                continue
            if self.rng.random() < 0.65:
                candidate[agent_id] = min(alternatives, key=lambda target_id: self._arrival_cost(agent_id, target_id))
            else:
                candidate[agent_id] = int(self.rng.choice(alternatives))
        return self._repair_assignment_capacity(candidate, active_target_ids)

    def _try_cs_structure_improvement(self, assignment, active_target_ids):
        incumbent_balls = self.leaf_balls
        incumbent_assignment = dict(assignment)
        incumbent_priority = self._assignment_priority(incumbent_assignment, incumbent_balls)

        for _ in range(max(2, self.cs_iterations)):
            proposal = self._clone_partition(incumbent_balls)
            if not proposal:
                break
            if self.rng.random() < 0.65:
                splittable = [ball for ball in proposal if len(ball.agent_ids) >= 2 * max(2, MIN_CAPTURE_AGENTS)]
                if not splittable:
                    continue
                chosen = min(splittable, key=lambda ball: ball.quality)
                children = self._split_candidate(chosen)
                if children is None:
                    continue
                proposal = [ball for ball in proposal if ball is not chosen] + children
            else:
                merged = self._try_merge_pair(proposal, len(active_target_ids))
                if merged is None:
                    continue
                proposal = merged

            proposal_assignment = self._capacity_aware_assignment(active_target_ids, proposal)
            proposal_priority = self._assignment_priority(proposal_assignment, proposal)
            if self._is_priority_better(proposal_priority, incumbent_priority):
                incumbent_balls = proposal
                incumbent_assignment = dict(proposal_assignment)
                incumbent_priority = proposal_priority
                self.metrics["recluster_count"] += 1
                self.metrics['cs_structure_accepts'] += 1

        self.leaf_balls = incumbent_balls
        self._refresh_tree()
        return incumbent_assignment, self._composite_cost(incumbent_assignment, incumbent_balls)

    def _try_cs_assignment_improvement(self, assignment, active_target_ids):
        incumbent = dict(assignment)
        incumbent_priority = self._assignment_priority(incumbent, self.leaf_balls)
        nests = [dict(incumbent)]
        for _ in range(self.cs_nests - 1):
            nests.append(self._mutate_assignment(incumbent, active_target_ids, self.leaf_balls))

        for _ in range(self.cs_iterations):
            updated = []
            for nest in nests:
                candidate = self._mutate_assignment(nest, active_target_ids, self.leaf_balls)
                accepted, chosen, _ = self._accept_assignment_if_better(nest, candidate, self.leaf_balls)
                updated.append(chosen if accepted else dict(nest))
            nests = updated

            priorities = [self._assignment_priority(nest, self.leaf_balls) for nest in nests]
            abandon_count = max(1, int(np.ceil(self.cs_pa * len(nests))))
            worst_indices = sorted(range(len(nests)), key=lambda index: priorities[index], reverse=True)[:abandon_count]
            for index in worst_indices:
                if self.rng.random() < self.cs_pa:
                    nests[index] = self._mutate_assignment(incumbent, active_target_ids, self.leaf_balls)

            best_nest = min(nests, key=lambda nest: self._assignment_priority(nest, self.leaf_balls))
            best_priority = self._assignment_priority(best_nest, self.leaf_balls)
            if self._is_priority_better(best_priority, incumbent_priority):
                incumbent = dict(best_nest)
                incumbent_priority = best_priority
                self.metrics['cs_assignment_accepts'] += 1
        return incumbent, self._composite_cost(incumbent, self.leaf_balls)

    def _plan_bounds_clip(self, params):
        clipped = np.asarray(params, dtype=float).copy()
        clipped[0] = clipped[0] % (2 * np.pi)
        clipped[1] = np.clip(clipped[1], 0.45, 0.72)
        clipped[2] = np.clip(clipped[2], 0.25, 0.75)
        return clipped

    def _encirclement_should_activate(self, target_id, critical_distance):
        """共享基础围捕控制器的近距离激活/退出滞回。"""
        target_id = int(target_id)
        if CAPTURE_MODE != 'geometric' or not USE_BASE_ENCIRCLEMENT_CONTROLLER:
            self.encirclement_active_targets.discard(target_id)
            return False

        was_active = target_id in self.encirclement_active_targets
        threshold = SFLA_DEACTIVATION_DISTANCE if was_active else SFLA_ACTIVATION_DISTANCE
        active = float(critical_distance) <= float(threshold)
        if active and not was_active:
            self.encirclement_active_targets.add(target_id)
            self.metrics['base_encirclement_activation_count'] += 1
            if self.use_sfla:
                self.metrics['sfla_activation_count'] += 1
        elif not active:
            self.encirclement_active_targets.discard(target_id)
        return active

    def _sfla_should_activate(self, target_id, critical_distance):
        """兼容旧接口；SFLA与基础控制器共享相同的近距离激活状态。"""
        return self._encirclement_should_activate(target_id, critical_distance)

    def _base_plan_parameters(self, agent_ids, target_id):
        """构造确定性的基础环形方案 ``[phase, radius_ratio, height_ratio]``。"""
        agent_ids = [int(agent_id) for agent_id in agent_ids]
        target_id = int(target_id)
        target_pos = self._target_reference_position(target_id)
        agent_map = self._active_agent_map()
        positions = np.asarray(
            [np.asarray(agent_map[agent_id][:3], dtype=float) for agent_id in agent_ids],
            dtype=float,
        )
        if len(positions):
            centroid = np.mean(positions[:, :2], axis=0)
            direction = centroid - target_pos[:2]
            if float(np.linalg.norm(direction)) > 1e-9:
                phase = float(np.mod(np.arctan2(direction[1], direction[0]), 2 * np.pi))
            else:
                phase = float(self.previous_sfla_plans.get(target_id, np.array([0.0]))[0])
        else:
            phase = 0.0
        return self._plan_bounds_clip(np.array([
            phase,
            BASE_RING_RADIUS_RATIO,
            BASE_RING_UAV_HEIGHT_RATIO,
        ], dtype=float))

    def _build_base_encirclement_waypoints(self, assignments, store_history=True):
        """为所有 GB 消融算法生成共享的远追击/近环形基础航点。"""
        active_agent_ids = {int(agent_id) for agent_id in assignments}
        active_target_ids = {int(target_id) for target_id in assignments.values()}
        self.previous_slot_waypoints = {
            int(agent_id): np.asarray(waypoint, dtype=float)
            for agent_id, waypoint in self.previous_slot_waypoints.items()
            if int(agent_id) in active_agent_ids
        }
        self.encirclement_active_targets.intersection_update(active_target_ids)

        grouped = {}
        for agent_id, target_id in assignments.items():
            grouped.setdefault(int(target_id), []).append(int(agent_id))

        waypoints = {}
        agent_map = self._active_agent_map()
        for target_id, agent_ids in grouped.items():
            target_ref = self._target_reference_position(target_id)
            distances = sorted(
                float(np.linalg.norm(np.asarray(agent_map[agent_id][:3]) - target_ref))
                for agent_id in agent_ids
            )
            required = min(max(1, int(MIN_CAPTURE_AGENTS)), len(distances))
            critical_distance = distances[required - 1] if distances else float('inf')

            if not self._encirclement_should_activate(target_id, critical_distance):
                for agent_id in agent_ids:
                    waypoint = np.asarray(target_ref, dtype=float).copy()
                    if int(agent_map[agent_id][6]) == 1:
                        waypoint[2] = USV_Z
                    waypoints[int(agent_id)] = waypoint
                continue

            base_params = self._base_plan_parameters(agent_ids, target_id)
            waypoints.update(
                self._plan_to_waypoints(
                    agent_ids, target_id, base_params, store_history=False
                )
            )
            self.metrics['base_encirclement_waypoint_steps'] += 1

        if store_history:
            for agent_id, waypoint in waypoints.items():
                self.previous_slot_waypoints[int(agent_id)] = np.asarray(waypoint, dtype=float).copy()
        return waypoints

    def _plan_to_waypoints(self, agent_ids, target_id, params, store_history=False):
        agent_ids = [int(agent_id) for agent_id in agent_ids]
        if not agent_ids:
            return {}
        params = self._plan_bounds_clip(params)
        target_pos = self._target_reference_position(target_id)
        ring_radius = float(params[1] * CAPTURE_RADIUS)
        angles = params[0] + 2 * np.pi * np.arange(len(agent_ids)) / len(agent_ids)
        agent_map = self._active_agent_map()

        slots_by_agent = {}
        cost_matrix = np.zeros((len(agent_ids), len(agent_ids)), dtype=float)
        for row, agent_id in enumerate(agent_ids):
            agent = agent_map[agent_id]
            agent_type = int(agent[6])
            max_vertical = np.sqrt(max(CAPTURE_RADIUS ** 2 - ring_radius ** 2, 0.0))
            z_offset = 0.0 if agent_type == 1 else float(params[2] * 0.80 * max_vertical)
            slots = np.column_stack([
                target_pos[0] + ring_radius * np.cos(angles),
                target_pos[1] + ring_radius * np.sin(angles),
                np.full(len(angles), USV_Z if agent_type == 1 else target_pos[2] + z_offset),
            ])
            slots_by_agent[agent_id] = slots
            arrival_cost = np.linalg.norm(slots - np.asarray(agent[:3]), axis=1) / max(self._max_speed(agent_id), 1e-9)
            previous_slot = self.previous_slot_waypoints.get(agent_id)
            if previous_slot is not None:
                continuity_cost = np.linalg.norm(slots - np.asarray(previous_slot), axis=1) / max(CAPTURE_RADIUS, 1e-9)
                arrival_cost = arrival_cost + self.sfla_slot_continuity_weight * continuity_cost
            cost_matrix[row] = arrival_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        waypoints = {}
        for row, col in zip(row_ind, col_ind):
            agent_id = agent_ids[int(row)]
            waypoint = np.asarray(slots_by_agent[agent_id][int(col)], dtype=float)
            delta = waypoint - target_pos
            distance = float(np.linalg.norm(delta))
            if distance > CAPTURE_RADIUS:
                waypoint = target_pos + delta * ((CAPTURE_RADIUS * 0.98) / max(distance, 1e-12))
            if int(agent_map[agent_id][6]) == 1:
                waypoint[2] = USV_Z
            waypoints[agent_id] = waypoint

        if store_history:
            for agent_id, waypoint in waypoints.items():
                self.previous_slot_waypoints[int(agent_id)] = np.asarray(waypoint, dtype=float).copy()
        return waypoints

    def _local_plan_cost(self, agent_ids, target_id, params):
        waypoints = self._plan_to_waypoints(agent_ids, target_id, params, store_history=False)
        if not waypoints:
            return float("inf")
        agent_map = self._active_agent_map()
        target_pos = self._target_reference_position(target_id)
        arrivals = []
        angles = []
        radii = []
        positions = []
        continuity_terms = []
        for agent_id, waypoint in waypoints.items():
            arrivals.append(np.linalg.norm(waypoint - np.asarray(agent_map[agent_id][:3])) / max(self._max_speed(agent_id), 1e-9))
            vector = waypoint[:2] - target_pos[:2]
            angles.append(float(np.mod(np.arctan2(vector[1], vector[0]), 2 * np.pi)))
            radii.append(float(np.linalg.norm(waypoint - target_pos)))
            positions.append(waypoint)
            previous_slot = self.previous_slot_waypoints.get(int(agent_id))
            if previous_slot is not None:
                continuity_terms.append(float(np.linalg.norm(waypoint - previous_slot) / max(CAPTURE_RADIUS, 1e-9)))

        sorted_angles = np.sort(np.asarray(angles))
        gaps = np.diff(np.r_[sorted_angles, sorted_angles[0] + 2 * np.pi]) if len(sorted_angles) > 1 else np.array([2 * np.pi])
        ideal_gap = 2 * np.pi / max(len(sorted_angles), 1)
        angular_error = float(np.max(np.abs(gaps - ideal_gap)) / (2 * np.pi))
        radius_error = float(np.mean(np.abs(np.asarray(radii) - np.mean(radii))) / max(CAPTURE_RADIUS, 1))

        collision_penalty = 0.0
        positions = np.asarray(positions)
        if len(positions) > 1:
            pairwise = cdist(positions, positions)
            pairwise += np.eye(len(positions)) * 1e9
            minimum_spacing = CAPTURE_RADIUS * 0.25
            collision_penalty = float(np.mean(np.maximum(0.0, minimum_spacing - pairwise) / minimum_spacing))

        arrival_norm = float(np.mean(arrivals) / max(ARENA_SIZE_XY / V_MAX_UAV, 1e-9))
        slot_continuity = float(np.mean(continuity_terms)) if continuity_terms else 0.0
        return float(
            4.0 * arrival_norm
            + 2.0 * angular_error
            + radius_error
            + 2.0 * collision_penalty
            + self.sfla_slot_continuity_weight * slot_continuity
        )

    def _sfla_target_waypoints(self, agent_ids, target_id, base_params=None):
        if not agent_ids:
            return {}
        target_id = int(target_id)
        base_candidate = self._plan_bounds_clip(
            self._base_plan_parameters(agent_ids, target_id)
            if base_params is None else np.asarray(base_params, dtype=float)
        )
        has_warm_start = target_id in self.previous_sfla_plans
        if has_warm_start:
            self.metrics['sfla_warm_start_hits'] += 1

        # 基础方案始终作为一只青蛙进入种群，确保 SFLA 在同一控制基线上优化。
        population = [base_candidate.copy()]
        if has_warm_start and self.sfla_population_size > 1:
            warm_start = self._plan_bounds_clip(
                np.asarray(self.previous_sfla_plans[target_id], dtype=float)
            )
            if not np.allclose(warm_start, base_candidate):
                population.append(warm_start)
        while len(population) < self.sfla_population_size:
            params = np.array([
                self.rng.uniform(0, 2 * np.pi),
                self.rng.uniform(0.45, 0.72),
                self.rng.uniform(0.25, 0.75),
            ])
            population.append(self._plan_bounds_clip(params))

        best_seen = float('inf')
        no_improvement = 0
        for _ in range(self.sfla_local_iterations):
            costs = np.asarray([
                self._local_plan_cost(agent_ids, target_id, params)
                for params in population
            ])
            order = np.argsort(costs)
            current_best = float(costs[int(order[0])])
            if best_seen - current_best > self.sfla_early_stop_tolerance:
                best_seen = current_best
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= self.sfla_early_stop_patience:
                    self.metrics['sfla_early_stops'] += 1
                    break

            global_best = population[int(order[0])].copy()
            memeplexes = [
                order[index::self.sfla_memeplex_count].tolist()
                for index in range(self.sfla_memeplex_count)
            ]
            for memeplex in memeplexes:
                if len(memeplex) < 2:
                    continue
                local_best_index = min(memeplex, key=lambda index: costs[index])
                worst_index = max(memeplex, key=lambda index: costs[index])
                worst = population[worst_index]
                local_best = population[local_best_index]

                candidate = self._plan_bounds_clip(
                    worst + self.rng.random(3) * (local_best - worst)
                )
                candidate_cost = self._local_plan_cost(agent_ids, target_id, candidate)
                if candidate_cost >= costs[worst_index]:
                    candidate = self._plan_bounds_clip(
                        worst + self.rng.random(3) * (global_best - worst)
                    )
                    candidate_cost = self._local_plan_cost(agent_ids, target_id, candidate)
                if candidate_cost >= costs[worst_index]:
                    candidate = self._plan_bounds_clip(np.array([
                        self.rng.uniform(0, 2 * np.pi),
                        self.rng.uniform(0.45, 0.72),
                        self.rng.uniform(0.25, 0.75),
                    ]))
                    candidate_cost = self._local_plan_cost(agent_ids, target_id, candidate)
                if candidate_cost < costs[worst_index]:
                    population[worst_index] = candidate

        best = min(
            population,
            key=lambda params: self._local_plan_cost(agent_ids, target_id, params),
        )
        self.previous_sfla_plans[target_id] = np.asarray(best, dtype=float).copy()
        return self._plan_to_waypoints(agent_ids, target_id, best, store_history=True)

    def _build_sfla_waypoints(self, assignments, base_waypoints=None):
        """在共享基础航点上，仅对已进入近距离的目标执行 SFLA 优化。"""
        if not self.use_sfla:
            return dict(base_waypoints or {})

        active_agent_ids = {int(agent_id) for agent_id in assignments}
        active_target_ids = {int(target_id) for target_id in assignments.values()}
        self.previous_slot_waypoints = {
            int(agent_id): np.asarray(waypoint, dtype=float)
            for agent_id, waypoint in self.previous_slot_waypoints.items()
            if int(agent_id) in active_agent_ids
        }
        self.previous_sfla_plans = {
            int(target_id): np.asarray(plan, dtype=float)
            for target_id, plan in self.previous_sfla_plans.items()
            if int(target_id) in active_target_ids
        }
        self.encirclement_active_targets.intersection_update(active_target_ids)

        grouped = {}
        for agent_id, target_id in assignments.items():
            grouped.setdefault(int(target_id), []).append(int(agent_id))

        waypoints = dict(base_waypoints or {})
        agent_map = self._active_agent_map()
        for target_id, agent_ids in grouped.items():
            target_ref = self._target_reference_position(target_id)
            distances = sorted(
                float(np.linalg.norm(np.asarray(agent_map[agent_id][:3]) - target_ref))
                for agent_id in agent_ids
            )
            required = min(max(1, int(MIN_CAPTURE_AGENTS)), len(distances))
            critical_distance = distances[required - 1] if distances else float('inf')

            if not self._encirclement_should_activate(target_id, critical_distance):
                continue

            base_params = self._base_plan_parameters(agent_ids, target_id)
            waypoints.update(
                self._sfla_target_waypoints(
                    agent_ids, target_id, base_params=base_params
                )
            )
            self.metrics['sfla_waypoint_steps'] += 1

        for agent_id, waypoint in waypoints.items():
            self.previous_slot_waypoints[int(agent_id)] = np.asarray(waypoint, dtype=float).copy()
        return waypoints

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def _get_all_leaves(self, node):
        if node is None:
            return []
        if node.is_leaf:
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self._get_all_leaves(child))
        return leaves

    def step(self):
        self._begin_step_cache()
        active_target_ids = self._active_target_ids()
        if not active_target_ids or not self._active_agent_map():
            self.desired_waypoints = {}
            return {}

        if self.cs_cooldown_remaining > 0:
            self.cs_cooldown_remaining -= 1

        self._sync_real_balls()
        self._adapt_structure(len(active_target_ids))

        assignment = self._capacity_aware_assignment(active_target_ids, self.leaf_balls)
        base_priority = self._assignment_priority(assignment, self.leaf_balls)

        if self._should_run_cs(base_priority, active_target_ids):
            self.metrics['cs_trigger_count'] += 1
            assignment, _ = self._try_cs_structure_improvement(assignment, active_target_ids)
            assignment, _ = self._try_cs_assignment_improvement(assignment, active_target_ids)
            self.cs_cooldown_remaining = CS_COOLDOWN

        assignment = self._repair_assignment_capacity(assignment, active_target_ids)
        assignment = self._stabilize_assignment(assignment, active_target_ids)
        final_priority = self._assignment_priority(assignment, self.leaf_balls)
        final_components = self._assignment_components(assignment, self.leaf_balls)
        objective = self._composite_cost(assignment, self.leaf_balls)
        self._update_cs_progress(final_priority)

        base_waypoints = self._build_base_encirclement_waypoints(
            assignment, store_history=not self.use_sfla
        )
        self.desired_waypoints = (
            self._build_sfla_waypoints(assignment, base_waypoints=base_waypoints)
            if self.use_sfla
            else base_waypoints
        )
        self.last_assignments = dict(assignment)
        self.last_objective = float(objective)
        self.metrics['assignment_makespan'] = float(final_components['makespan'])
        self.metrics['assignment_completion_gap'] = float(final_components['completion'])
        self.metrics['assignment_angular_gap'] = float(final_components['angular'])
        self._refresh_tree()

        print(
            f"   Assignments ({self.name}, SFLA={self.use_sfla}, CS={self.use_cs}): "
            f"{assignment}; balls={len(self.leaf_balls)}, objective={self.last_objective:.4f}, "
            f"makespan={final_components['makespan']:.4f}, "
            f"completion_gap={final_components['completion']:.1f}"
        )
        return assignment


# ==================== 统一主类上的粒球消融包装器 ====================
class GBSFLAAlgorithm(GBSFLACSAlgorithm):
    """GB + SFLA：与完整算法共用所有流程，仅关闭 CS。"""

    def __init__(self, env, name="GB-SFLA"):
        super().__init__(env, name=name, use_sfla=True, use_cs=False)


class GBCSAlgorithm(GBSFLACSAlgorithm):
    """GB + CS：使用共享基础环形控制，仅关闭 SFLA 航点优化。"""

    def __init__(self, env, name="GB-CS"):
        super().__init__(env, name=name, use_sfla=False, use_cs=True)


class GBOnlyAlgorithm(GBSFLACSAlgorithm):
    """GB-only：使用共享基础环形控制，同时关闭 SFLA 优化和 CS。"""

    def __init__(self, env, name="GB-only"):
        super().__init__(env, name=name, use_sfla=False, use_cs=False)


# ==================== 消融算法C：CS-only（仅CS，无粒球、无SFLA） ====================
class CSOnlyAlgorithm(BaseAlgorithm):
    """GB-CS-SFLA 的消融版本：不使用粒球分组，直接用 CS 优化个体到目标的分配。"""

    def __init__(self, env, name="CS-only"):
        super().__init__(env, name)
        self.n_nests = 40
        self.pa = CS_PA
        self.beta = CS_BETA
        self.local_iterations = 5
        self.nests = None
        self.best_solution = None
        self.best_fitness = -np.inf
        self._last_active_targets = None

    def _active_target_ids(self):
        return [j for j in range(len(self.env.targets)) if j not in self.env.permanently_captured]

    def _random_solution(self, active_targets):
        if not active_targets:
            return np.array([], dtype=int)
        return np.random.choice(active_targets, size=len(self.env.agents)).astype(int)

    def _initialize_nests(self, active_targets):
        self.nests = [self._random_solution(active_targets) for _ in range(self.n_nests)]
        self.best_solution = None
        self.best_fitness = -np.inf
        for solution in self.nests:
            fitness = self._fitness(solution, active_targets)
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution.copy()

    def _repair_solution(self, solution, active_targets):
        """目标被捕获后，将失效目标编号替换为当前活跃目标。"""
        if not active_targets:
            return np.array([], dtype=int)
        repaired = np.asarray(solution, dtype=int).copy()
        if len(repaired) != len(self.env.agents):
            return self._random_solution(active_targets)
        active_set = set(active_targets)
        invalid_indices = [idx for idx, target_id in enumerate(repaired) if int(target_id) not in active_set]
        if invalid_indices:
            repaired[invalid_indices] = np.random.choice(active_targets, size=len(invalid_indices))
        return repaired

    def _fitness(self, solution, active_targets):
        if len(solution) != len(self.env.agents) or not active_targets:
            return 0.0

        total_distance = 0.0
        for i, agent in enumerate(self.env.agents):
            target_id = int(solution[i])
            if target_id not in active_targets:
                total_distance += ARENA_SIZE_XY * 10
                continue
            total_distance += np.linalg.norm(agent[:3] - self.env.targets[target_id, :3])
        return 1.0 / (total_distance + 1e-5)

    def _levy_flight(self, beta=None):
        if beta is None:
            beta = self.beta
        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                   (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, 1)
        v = np.random.normal(0, 1, 1)
        return float(u / (abs(v) ** (1 / beta)))

    def _generate_cuckoo(self, solution, active_targets):
        """用离散莱维扰动产生新巢：部分智能体继承当前最优目标，其余随机探索。"""
        candidate = self._repair_solution(solution, active_targets)
        if len(candidate) == 0:
            return candidate

        levy_step = abs(self._levy_flight())
        mutation_rate = min(0.5, max(1.0 / len(candidate), 0.05 * levy_step))
        mutation_count = max(1, int(np.ceil(mutation_rate * len(candidate))))
        mutation_indices = np.random.choice(len(candidate), size=mutation_count, replace=False)

        for idx in mutation_indices:
            if self.best_solution is not None and np.random.random() < 0.5:
                candidate[idx] = self.best_solution[idx]
            else:
                candidate[idx] = np.random.choice(active_targets)
        return candidate

    def step(self):
        active_targets = self._active_target_ids()
        if not active_targets:
            return {}

        active_key = tuple(active_targets)
        if self.nests is None or self._last_active_targets != active_key:
            self._initialize_nests(active_targets)
            self._last_active_targets = active_key
        else:
            self.nests = [self._repair_solution(solution, active_targets) for solution in self.nests]
            self.best_solution = self._repair_solution(self.best_solution, active_targets)
            self.best_fitness = self._fitness(self.best_solution, active_targets)

        for _ in range(self.local_iterations):
            # 莱维飞行生成新解，若更优则替换当前巢。
            for i, solution in enumerate(self.nests):
                candidate = self._generate_cuckoo(solution, active_targets)
                if self._fitness(candidate, active_targets) > self._fitness(solution, active_targets):
                    self.nests[i] = candidate

            # 发现概率 pa：替换适应度较差的一部分巢。
            fitness_values = np.array([self._fitness(solution, active_targets) for solution in self.nests])
            abandon_count = max(1, int(self.pa * self.n_nests))
            worst_indices = np.argsort(fitness_values)[:abandon_count]
            for idx in worst_indices:
                if np.random.random() < self.pa:
                    self.nests[idx] = self._random_solution(active_targets)

            fitness_values = np.array([self._fitness(solution, active_targets) for solution in self.nests])
            best_idx = int(np.argmax(fitness_values))
            if fitness_values[best_idx] > self.best_fitness:
                self.best_fitness = float(fitness_values[best_idx])
                self.best_solution = self.nests[best_idx].copy()

        self.metrics["recluster_count"] += 1

        assignments = {}
        if self.best_solution is not None:
            for i, agent in enumerate(self.env.agents):
                assignments[int(agent[7])] = int(self.best_solution[i])

        print(f"   Assignments (CS-only ablation): {assignments}")
        return assignments



# ==================== 消融算法D：KMeans-grouping（KMeans分组，无粒球、无SFLA、无CS） ====================
class KMeansGroupingAlgorithm(BaseAlgorithm):
    """不用粒球，直接采用 KMeans 对智能体进行分组；k 由当前未捕获目标数确定。"""

    class KMeansCluster:
        """仅用于记录 KMeans 分组结果和可视化，不属于粒球结构。"""

        def __init__(self, cluster_id, points, agent_ids):
            self.ball_id = cluster_id  # 兼容原有绘图代码中的字段命名
            self.cluster_id = cluster_id
            self.points = np.asarray(points, dtype=float) if len(points) > 0 else np.empty((0, 3))
            self.agent_ids = list(agent_ids)
            self.children = []
            self.is_leaf = True
            if len(self.points) > 0:
                self.center = np.mean(self.points, axis=0)
                self.radius = float(np.max(cdist([self.center], self.points))) if len(self.points) > 0 else 0.0
            else:
                self.center = np.zeros(3)
                self.radius = 0.0

    def __init__(self, env, name="KMeans-grouping"):
        super().__init__(env, name)
        self.random_state = SEED
        self.clusters_for_drawing = []

    def _active_target_ids(self):
        return [j for j in range(len(self.env.targets)) if j not in self.env.permanently_captured]

    def _active_agents(self):
        guarding_ids = set()
        if hasattr(self.env, '_get_guarding_agent_ids'):
            guarding_ids = self.env._get_guarding_agent_ids()
        return [agent for agent in self.env.agents if int(agent[7]) not in guarding_ids]

    def _build_clusters(self, active_agents, k):
        """执行 KMeans 分组，并返回包含中心、半径和成员 ID 的簇对象。"""
        positions = np.asarray([agent[:3] for agent in active_agents], dtype=float)
        agent_ids = [int(agent[7]) for agent in active_agents]

        if len(active_agents) == 0 or k <= 0:
            return []

        # KMeans 要求簇数不能超过样本数；当位置完全重复时，进一步限制到唯一位置数，避免空簇警告。
        unique_position_count = len(np.unique(np.round(positions, decimals=8), axis=0))
        k = max(1, min(k, len(active_agents), unique_position_count))

        if k == 1:
            labels = np.zeros(len(active_agents), dtype=int)
        else:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(positions)

        clusters = []
        for cluster_id in range(k):
            indices = np.where(labels == cluster_id)[0]
            if len(indices) == 0:
                continue
            cluster_points = positions[indices]
            cluster_agent_ids = [agent_ids[idx] for idx in indices]
            clusters.append(self.KMeansCluster(cluster_id, cluster_points, cluster_agent_ids))
        return clusters

    def step(self):
        """KMeans 分组 → 组中心到目标的匈牙利匹配 → 展开为 {智能体ID: 目标ID}。"""
        assignments = {}
        active_targets = self._active_target_ids()
        active_agents = self._active_agents()

        if not active_targets or not active_agents:
            self.clusters_for_drawing = []
            return assignments

        # k 由目标数确定；考虑已捕获目标后，采用当前未捕获目标数。
        k = len(active_targets)
        clusters = self._build_clusters(active_agents, k)
        self.clusters_for_drawing = clusters
        self.metrics["recluster_count"] += 1

        if not clusters:
            return assignments

        cost_matrix = np.zeros((len(clusters), len(active_targets)))
        for i, cluster in enumerate(clusters):
            for j, target_id in enumerate(active_targets):
                target_pos = self.env.targets[target_id, :3]
                cost_matrix[i, j] = np.linalg.norm(cluster.center - target_pos)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        cluster_to_target = {
            int(cluster_idx): int(active_targets[target_idx])
            for cluster_idx, target_idx in zip(row_ind, col_ind)
        }

        for cluster_idx, target_id in cluster_to_target.items():
            for agent_id in clusters[cluster_idx].agent_ids:
                assignments[agent_id] = target_id

        print(f"   Assignments (KMeans-grouping, k={len(active_targets)}): {assignments}")
        return assignments

# ==================== 算法2：标准SFLA（无粒球引导） ====================
class StandardSFLAAlgorithm(BaseAlgorithm):
    """标准蛙跳算法（对比基线）"""

    def __init__(self, env, name="Standard-SFLA"):
        super().__init__(env, name)
        self.population_size = 50
        self.memeplex_count = 5
        self.local_iterations = 10
        self.max_step = 50
        self.population = None

    def _initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            solution = np.random.randint(0, len(self.env.targets), size=len(self.env.agents))
            self.population.append(solution)

    def _evaluate_fitness(self, solution):
        """评估适应度（总距离最小化）"""
        total_distance = 0
        for i, agent in enumerate(self.env.agents):
            target_id = solution[i]
            if target_id < len(self.env.targets):
                target_pos = self.env.targets[target_id, :3]
                agent_pos = agent[:3]
                total_distance += np.linalg.norm(target_pos - agent_pos)
        return 1 / (total_distance + 1e-5)

    def _local_search(self, memeplex):
        """局部搜索"""
        for _ in range(self.local_iterations):
            fitness = [self._evaluate_fitness(sol) for sol in memeplex]
            best_idx = np.argmax(fitness)
            worst_idx = np.argmin(fitness)

            best_sol = memeplex[best_idx]
            worst_sol = memeplex[worst_idx].copy()

            for i in range(len(worst_sol)):
                if np.random.random() < 0.5:
                    worst_sol[i] = best_sol[i]
                else:
                    worst_sol[i] = (worst_sol[i] + np.random.randint(-1, 2)) % len(self.env.targets)

            memeplex[worst_idx] = worst_sol

        return memeplex

    def step(self):
        """执行一步标准SFLA"""
        if self.population is None:
            self._initialize_population()

        memeplex_size = self.population_size // self.memeplex_count
        memeplexes = []
        for i in range(self.memeplex_count):
            start = i * memeplex_size
            end = start + memeplex_size if i < self.memeplex_count - 1 else self.population_size
            memeplexes.append(self.population[start:end])

        new_population = []
        for memeplex in memeplexes:
            updated_memeplex = self._local_search(memeplex)
            new_population.extend(updated_memeplex)

        self.population = new_population
        self.metrics["recluster_count"] += 1

        best_solution = max(self.population, key=lambda x: self._evaluate_fitness(x))
        assignments = {}
        for i, agent in enumerate(self.env.agents):
            agent_id = int(agent[7])
            target_id = best_solution[i]
            assignments[agent_id] = target_id

        return assignments


# ==================== 算法3：PSO算法 ====================
class PSOAlgorithm(BaseAlgorithm):
    """粒子群优化算法（对比基线）"""

    def __init__(self, env, name="PSO"):
        super().__init__(env, name)
        self.num_particles = 50
        self.w = 0.8
        self.c1 = 1.5
        self.c2 = 1.5
        self.particles = None
        self.velocities = None
        self.pbest = None
        self.gbest = None

    def _initialize_swarm(self):
        """初始化粒子群"""
        self.particles = []
        self.velocities = []
        self.pbest = []

        for _ in range(self.num_particles):
            particle = np.random.randint(0, len(self.env.targets), size=len(self.env.agents))
            velocity = np.random.uniform(-1, 1, size=len(self.env.agents))

            self.particles.append(particle)
            self.velocities.append(velocity)
            self.pbest.append(particle.copy())

        self.gbest = self.particles[0].copy()

    def _evaluate_fitness(self, particle):
        """评估适应度"""
        total_distance = 0
        for i, agent in enumerate(self.env.agents):
            target_id = particle[i]
            if target_id < len(self.env.targets):
                target_pos = self.env.targets[target_id, :3]
                agent_pos = agent[:3]
                total_distance += np.linalg.norm(target_pos - agent_pos)
        return 1 / (total_distance + 1e-5)

    def step(self):
        """执行一步PSO"""
        if self.particles is None:
            self._initialize_swarm()

        for i in range(self.num_particles):
            r1, r2 = np.random.random(), np.random.random()
            cognitive = self.c1 * r1 * (self.pbest[i] - self.particles[i])
            social = self.c2 * r2 * (self.gbest - self.particles[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive + social

            self.particles[i] = self.particles[i] + self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], 0, len(self.env.targets) - 1).astype(int)

            if self._evaluate_fitness(self.particles[i]) > self._evaluate_fitness(self.pbest[i]):
                self.pbest[i] = self.particles[i].copy()

        for particle in self.particles:
            if self._evaluate_fitness(particle) > self._evaluate_fitness(self.gbest):
                self.gbest = particle.copy()

        self.metrics["recluster_count"] += 1

        assignments = {}
        for i, agent in enumerate(self.env.agents):
            target_id = self.gbest[i]
            if target_id < len(self.env.targets):
                assignments[int(agent[7])] = target_id

        return assignments


# ==================== 算法4：GA算法 ====================
class GAAlgorithm(BaseAlgorithm):
    """遗传算法（对比基线）"""

    def __init__(self, env, name="GA"):
        super().__init__(env, name)
        self.population_size = 50
        self.elite_size = 10
        self.mutation_rate = 0.1
        self.population = None

    def _initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            chromosome = np.random.randint(0, len(self.env.targets), size=len(self.env.agents))
            self.population.append(chromosome)

    def _fitness_function(self, chromosome):
        """适应度函数"""
        total_distance = 0
        for i, agent in enumerate(self.env.agents):
            target_id = chromosome[i]
            if target_id < len(self.env.targets):
                target_pos = self.env.targets[target_id, :3]
                agent_pos = agent[:3]
                total_distance += np.linalg.norm(target_pos - agent_pos)
        return 1 / (total_distance + 1e-5)

    def _selection(self):
        """选择操作"""
        fitness_values = np.array([self._fitness_function(chrom) for chrom in self.population])
        fitness_sum = np.sum(fitness_values)

        if fitness_sum == 0:
            probabilities = np.ones(len(fitness_values)) / len(fitness_values)
        else:
            probabilities = fitness_values / fitness_sum

        selected_indices = np.random.choice(
            len(self.population),
            size=len(self.population) - self.elite_size,
            p=probabilities
        )
        return [self.population[i] for i in selected_indices]

    def _crossover(self, parent1, parent2):
        """交叉操作"""
        if len(parent1) != len(parent2):
            return parent1.copy()

        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2

    def _mutation(self, chromosome):
        """变异操作"""
        mutated = chromosome.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] = np.random.randint(0, len(self.env.targets))
        return mutated

    def step(self):
        """执行一步GA"""
        if self.population is None:
            self._initialize_population()

        elite_indices = np.argsort([self._fitness_function(chrom) for chrom in self.population])[-self.elite_size:]
        elites = [self.population[i] for i in elite_indices]

        selected = self._selection()

        new_population = elites.copy()
        for i in range(0, len(selected) - 1, 2):
            if i + 1 < len(selected):
                child1, child2 = self._crossover(selected[i], selected[i + 1])
                new_population.append(self._mutation(child1))
                new_population.append(self._mutation(child2))

        while len(new_population) < self.population_size:
            new_population.append(self._mutation(selected[np.random.randint(len(selected))]))

        self.population = new_population[:self.population_size]
        self.metrics["recluster_count"] += 1

        best_chromosome = max(self.population, key=lambda x: self._fitness_function(x))
        assignments = {}
        for i, agent in enumerate(self.env.agents):
            target_id = best_chromosome[i]
            if target_id < len(self.env.targets):
                assignments[int(agent[7])] = target_id

        return assignments



# ==================== 3. 数据结构：粒球节点 ====================
class GranularBallNode:
    def __init__(self, ball_id, points, parent=None):
        self.ball_id = ball_id
        self.points = np.array(points)
        self.agent_ids = []  #  新增：存储智能体ID
        self.parent = parent
        self.children = []
        self.children_quality = []  # ✅ 新增：存储子节点的质量值
        self.is_empty = False  # ✅ 新增：标记是否为空节点

        # 物理属性
        self.center = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)
        self.radius = np.max(cdist([self.center], points)) if len(points) > 0 else 0

        # 质量属性 (用于CS算法判断)
        self.quality = 0.0
        self.is_leaf = True

    def split(self, next_id_func, env_instance=None):
        n_points = len(self.points)
        if n_points < 4:
            return False

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.points)

        unique_labels, counts = np.unique(labels, return_counts=True)
        if np.any(counts < 2):
            return False

        new_children = []
        for i in range(2):
            child_points = self.points[labels == i]
            if len(child_points) < 2:
                return False
            child_node = GranularBallNode(ball_id=next_id_func(), points=child_points, parent=self)
            new_children.append(child_node)

        # ✅ 关键修复：检查子节点是否有智能体
        children_with_agents = []
        for child in new_children:
            # 检查子节点是否有智能体
            if hasattr(child, 'agent_ids') and child.agent_ids:
                children_with_agents.append(child)
            elif len(child.points) > 0:
                # 如果子节点有点但没有agent_ids，给它分配空的agent_ids
                child.agent_ids = []
                children_with_agents.append(child)

        # ✅ 如果只有一个子节点有智能体，父节点直接等于这个子节点
        if len(children_with_agents) == 1:
            only_child = children_with_agents[0]
            # 将父节点的属性设置为子节点的属性
            self.points = only_child.points
            self.agent_ids = only_child.agent_ids
            self.center = only_child.center
            self.radius = only_child.radius
            self.children = []  # 清空子节点
            self.is_leaf = True  # 变回叶子节点
            self.children_quality = []  # 清空子节点质量
            print(f"🔄 Ball-{self.ball_id} 吸收唯一有智能体的子节点")
            return True

        # ✅ 如果两个子节点都没有智能体，不进行分裂
        if len(children_with_agents) == 0:
            print(f"⚠️ Ball-{self.ball_id} 分裂失败：两个子节点都没有智能体")
            return False

        # ✅ 正常情况：两个子节点都有智能体
        self.children = new_children
        self.is_leaf = False

        # ✅ 存储子节点的质量值
        self.children_quality = []
        if env_instance is not None:
            for child in self.children:
                # 计算子节点的质量
                child.quality = env_instance._calculate_cluster_quality(child)
                self.children_quality.append(child.quality)

        return True


# ==================== 4. 环境主类 (SwarmEnv3D) ====================
class SwarmEnv3D:
    def __init__(self):
        self.guarding_agents = {}  # 键：目标ID → 值：守卫智能体ID集合
        self.reset()
        # -----------------------------------------------------------------------------
        self.algorithm = None
        self.algorithm_name = "GB-CS-SFLA"
        self.set_algorithm(self.algorithm_name)

    def set_algorithm(self, algorithm_name):
        self.algorithm_name = algorithm_name
        if algorithm_name == "GB-CS-SFLA":
            self.algorithm = GBSFLACSAlgorithm(self, "GB-CS-SFLA", use_sfla=True, use_cs=True)
        elif algorithm_name == "GB-SFLA":
            self.algorithm = GBSFLAAlgorithm(self, "GB-SFLA")
        elif algorithm_name == "GB-CS":
            self.algorithm = GBCSAlgorithm(self, "GB-CS")
        elif algorithm_name == "GB-only":
            self.algorithm = GBOnlyAlgorithm(self, "GB-only")
        elif algorithm_name == "CS-only":
            self.algorithm = CSOnlyAlgorithm(self, "CS-only")
        elif algorithm_name in ["KMeans-grouping", "KMeans"]:
            self.algorithm = KMeansGroupingAlgorithm(self, "KMeans-grouping")
        elif algorithm_name == "Standard-SFLA":
            self.algorithm = StandardSFLAAlgorithm(self, "Standard-SFLA")
        elif algorithm_name == "PSO":
            self.algorithm = PSOAlgorithm(self, "PSO")
        elif algorithm_name == "GA":
            self.algorithm = GAAlgorithm(self, "GA")
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")


    # -----------------------------------------------------------------------------

    def _calculate_repulsion_force(self, agent_pos, all_agents_pos, obstacles_pos=None):
        repulsion_force = np.zeros(3)
        for other_pos in all_agents_pos:
            if np.linalg.norm(agent_pos - other_pos) < 1e-5: continue
            dist_vec = agent_pos - other_pos
            dist = np.linalg.norm(dist_vec)
            if dist < self.repulsion_range:
                if dist == 0: dist = 0.01
                direction = dist_vec / dist
                force_magnitude = self.repulsion_gain * (1.0 / dist)
                repulsion_force += direction * force_magnitude
        return repulsion_force

    def reset(self):
        self.time_step = 0
        self.permanently_captured = set()
        self.total_travel_distance = 0.0
        self.target_load_balance_history = []
        self.reconfiguration_count = 0
        self.previous_assignments = {}
        self.guarding_agents = {}
        self.target_captors = {}
        self.released_agents_last_step = set()
        self.repulsion_range = 50
        self.repulsion_gain  = 100
        random.seed(SEED)
        np.random.seed(SEED)
        # --- 初始化智能体 ---
        self.agents = []

        # 根据 INIT_RANDOMSTATE 决定初始化方式
        if INIT_RANDOMSTATE == 1:
            # ===== 原有随机分布方式 =====
            print("🎲 使用随机分布初始化智能体")

            # UAV 随机分布
            for i in range(UAV_COUNT):
                pos_x = random.uniform(ARENA_SIZE_XY * 0.1, ARENA_SIZE_XY * 0.9)
                pos_y = random.uniform(ARENA_SIZE_XY * 0.1, ARENA_SIZE_XY * 0.9)
                pos_z = random.uniform(UAV_Z_MIN, UAV_Z_MAX * 0.5)
                vel = random.uniform(V_MIN, V_MAX_UAV)
                theta = random.uniform(0, 2 * np.pi)
                phi = random.uniform(-np.pi / 2, np.pi / 2)
                self.agents.append([pos_x, pos_y, pos_z, vel, theta, phi, 0, i + 1000])

            # USV 随机分布
            for i in range(USV_COUNT):
                pos_x = random.uniform(ARENA_SIZE_XY * 0.05, ARENA_SIZE_XY * 0.95)
                pos_y = random.uniform(ARENA_SIZE_XY * 0.05, ARENA_SIZE_XY * 0.95)
                pos_z = USV_Z
                vel = random.uniform(V_MIN, V_MAX_USV)
                theta = random.uniform(0, 2 * np.pi)
                phi = 0
                self.agents.append([pos_x, pos_y, pos_z, vel, theta, phi, 1, i + 2000])

        else:
            # ===== 新增：均匀分布方式 =====
            print("📐 使用均匀分布初始化智能体")

            # 计算网格布局
            total_agents = UAV_COUNT + USV_COUNT

            # 计算网格行列数（尽量接近正方形）
            grid_cols = int(np.ceil(np.sqrt(total_agents)))
            grid_rows = int(np.ceil(total_agents / grid_cols))

            # 计算每个网格的大小
            cell_width = ARENA_SIZE_XY / grid_cols
            cell_height = ARENA_SIZE_XY / grid_rows

            # 添加小随机扰动避免完全对齐（可选）
            jitter = 0.1  # 扰动幅度（相对于单元格大小）

            # 初始化所有智能体位置
            agent_positions = []
            agent_id = 0

            # 生成网格点
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if agent_id >= total_agents:
                        break

                    # 计算基础位置（网格中心）
                    base_x = col * cell_width + cell_width / 2
                    base_y = row * cell_height + cell_height / 2

                    # 添加小随机扰动
                    jitter_x = random.uniform(-cell_width * jitter, cell_width * jitter)
                    jitter_y = random.uniform(-cell_height * jitter, cell_height * jitter)

                    pos_x = base_x + jitter_x
                    pos_y = base_y + jitter_y

                    # 确保在边界内
                    pos_x = max(cell_width * 0.1, min(pos_x, ARENA_SIZE_XY - cell_width * 0.1))
                    pos_y = max(cell_height * 0.1, min(pos_y, ARENA_SIZE_XY - cell_height * 0.1))

                    agent_positions.append([pos_x, pos_y, agent_id])
                    agent_id += 1

            # 分配 UAV（前 UAV_COUNT 个位置给 UAV）
            # for i in range(UAV_COUNT + USV_COUNT):
            for i in range(UAV_COUNT):
                if i < len(agent_positions):
                    pos_x, pos_y, _ = agent_positions[i]
                    # UAV 在高度上均匀分布
                    pos_z = random.uniform(UAV_Z_MIN, UAV_Z_MAX)
                    vel = random.uniform(V_MIN, V_MAX_UAV)
                    theta = random.uniform(0, 2 * np.pi)
                    phi = random.uniform(-np.pi / 2, np.pi / 2)
                    self.agents.append([pos_x, pos_y, pos_z, vel, theta, phi, 0, i + 1000])

            # 分配 USV（剩余位置给 USV）
            usv_start_id = UAV_COUNT
            for i in range(USV_COUNT):
                if usv_start_id + i < len(agent_positions):
                    pos_x, pos_y, _ = agent_positions[usv_start_id + i]
                    pos_z = USV_Z
                    vel = random.uniform(V_MIN, V_MAX_USV)
                    theta = random.uniform(0, 2 * np.pi)
                    phi = 0
                    self.agents.append([pos_x, pos_y, pos_z, vel, theta, phi, 1, i + 2000])

        self.agents = np.array(self.agents, dtype=float)
        self._validate_unique_agent_ids()
        self.agent_index_by_id = {
            int(agent[7]): index
            for index, agent in enumerate(self.agents)
        }
        print("self.agents",self.agents)
        # --- 初始化目标（保持不变）---
        self.targets = []
        tar_num = 0
        for i in range(TARGET_COUNT):
            if tar_num:
                pos_x = random.uniform(ARENA_SIZE_XY * -0.2, ARENA_SIZE_XY * 1.25)
                pos_y = random.uniform(ARENA_SIZE_XY * -0.2, ARENA_SIZE_XY * 1.25)
            else:
                pos_x = random.uniform(ARENA_SIZE_XY * 0.2, ARENA_SIZE_XY * 0.8)
                pos_y = random.uniform(ARENA_SIZE_XY * 0.2, ARENA_SIZE_XY * 0.8)
            pos_z = TARGET_Z
            vel = random.uniform(TARGET_SPEED * 0.5, TARGET_SPEED)
            theta = random.uniform(0, 2 * np.pi)
            self.targets.append([pos_x, pos_y, pos_z, vel, theta, i + 3000])
        self.targets = np.array(self.targets)
        print('targets',self.targets)
        # --- 初始化粒球树（保持不变）---
        self.all_balls = {}
        self.next_ball_id = 0
        root_ball = GranularBallNode(self.next_ball_id, [], parent=None)
        self.ball_tree = root_ball
        self.all_balls[self.next_ball_id] = root_ball
        self.next_ball_id += 1


    def _get_guarding_agent_ids(self):
        """Return agents already locked into post-capture guard duties."""
        guarding_ids = set()
        for guard_ids in self.guarding_agents.values():
            guarding_ids.update(int(agent_id) for agent_id in guard_ids)
        return guarding_ids

    def _get_active_agents(self):
        """Return non-guarding agents used by GB grouping and task allocation."""
        guarding_ids = self._get_guarding_agent_ids()
        return [
            agent for agent in self.agents
            if int(agent[7]) not in guarding_ids
        ]

    def _validate_unique_agent_ids(self):
        """Fail fast if UAV/USV IDs overlap, because assignments use IDs as dict keys."""
        ids = [int(agent[7]) for agent in self.agents]
        if len(ids) != len(set(ids)):
            duplicates = sorted({agent_id for agent_id in ids if ids.count(agent_id) > 1})
            raise ValueError(f"Duplicate agent IDs detected: {duplicates}")

    def _sanitize_assignments(self, assignments):
        """Ensure every non-guarding agent is directed to an active target."""
        active_target_ids = [
            target_id for target_id in range(len(self.targets))
            if target_id not in self.permanently_captured
        ]
        if not active_target_ids:
            return {}

        sanitized = dict(assignments)
        guarding_ids = self._get_guarding_agent_ids()
        for agent in self.agents:
            agent_id = int(agent[7])
            if agent_id in guarding_ids:
                continue

            target_id = sanitized.get(agent_id)
            if target_id not in active_target_ids:
                distances = [
                    np.linalg.norm(agent[:3] - self.targets[candidate_id, :3])
                    for candidate_id in active_target_ids
                ]
                sanitized[agent_id] = active_target_ids[int(np.argmin(distances))]
        return sanitized

    def _record_assignment_metrics(self, assignments):
        """Record load balance and assignment changes under one shared definition."""
        active_target_ids = [
            target_id for target_id in range(len(self.targets))
            if target_id not in self.permanently_captured
        ]
        guarding_ids = self._get_guarding_agent_ids()
        active_agent_ids = {
            int(agent[7]) for agent in self.agents
            if int(agent[7]) not in guarding_ids
        }
        effective_assignments = {
            agent_id: target_id
            for agent_id, target_id in assignments.items()
            if agent_id in active_agent_ids and target_id in active_target_ids
        }

        load_balance = calculate_target_load_balance(
            effective_assignments,
            active_target_ids,
            active_agent_ids,
        )
        self.target_load_balance_history.append(load_balance)
        self.reconfiguration_count += count_reconfigurations(
            self.previous_assignments,
            effective_assignments,
            active_agent_ids,
        )
        self.previous_assignments = effective_assignments.copy()

        self.algorithm.metrics["reconfiguration_count"] = self.reconfiguration_count
        self.algorithm.metrics["target_load_balance"] = load_balance



    # ==================== 新增：莱维飞行 (CS核心) ====================
    # def _levy_flight(self, beta=CS_BETA):
    #     """生成莱维飞行步长"""
    #     sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
    #                (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    #     u = np.random.normal(0, sigma_u, 1)
    #     v = np.random.normal(0, 1, 1)
    #     step = u / (abs(v) ** (1 / beta))
    #     return step[0]

    # ==================== 新增：SFLA 局部搜索 ====================
    # def _sfla_local_search(self, node):
    #     """
    #     在粒球内部执行 SFLA。
    #     逻辑：将组内智能体视为青蛙，向组内最优位置（食物最丰富处）跳跃。
    #     """
    #     if len(node.points) < 2: return
    #
    #     # 保存原始 agent_ids
    #     original_ids = node.agent_ids.copy() if hasattr(node, 'agent_ids') else []
    #
    #     frogs = node.points.copy()
    #     n_frogs = len(frogs)
    #
    #     # 1. 寻找当前粒球内的全局最优位置 (即球心，或者距离中心最近的点)
    #     global_best_pos = node.center
    #
    #     new_positions = []
    #     for frog in frogs:
    #         # 2. 计算跳跃向量
    #         dist_vec = global_best_pos - frog
    #         dist = np.linalg.norm(dist_vec)
    #
    #         # 3. 计算步长 (受 SFLA_MAX_STEP 限制)
    #         if dist > 1e-5:
    #             jump_dist = min(dist * 0.5, SFLA_MAX_STEP)
    #             move_vec = (dist_vec / dist) * jump_dist
    #         else:
    #             move_vec = np.random.randn(3) * 10  # 已经在中心附近，随机微扰
    #
    #         new_pos = frog + move_vec
    #         new_positions.append(new_pos)
    #
    #     # 4. 更新节点位置
    #     node.points = np.array(new_positions)
    #     node.center = np.mean(node.points, axis=0)
    #
    #     # 5. 恢复 agent_ids（因为坐标变化了，但ID归属不变）
    #     node.agent_ids = original_ids

    # 保留 _sfla_local_search 方法（但简化版本）
    def _sfla_local_search(self, node):
        """简化的SFLA局部搜索"""
        if len(node.points) < 2:
            return

        # 简单的向中心移动
        center = node.center
        new_positions = []
        for point in node.points:
            direction = center - point
            dist = np.linalg.norm(direction)
            if dist > 1e-5:
                direction = direction / dist
            new_pos = point + direction * min(dist * 0.5, 50)
            new_positions.append(new_pos)

        node.points = np.array(new_positions)
        node.center = np.mean(node.points, axis=0)

    # ==================== 新增：CS 全局重组 ====================
    # def _cs_global_restructure(self, node):
    #     """
    #     利用 CS 机制重组质量差的粒球。
    #     逻辑：如果质量差，执行莱维飞行大跳跃，或者强制分裂。
    #     """
    #     # 保存原始 agent_ids
    #     original_ids = node.agent_ids.copy() if hasattr(node, 'agent_ids') else []
    #
    #     if random.random() < CS_PA:
    #         # --- 莱维飞行探索 ---
    #         step_size = self._levy_flight() * 100
    #         direction = np.random.randn(3)
    #         direction = direction / (np.linalg.norm(direction) + 1e-5)
    #
    #         node.center += direction * step_size
    #         if len(node.points) > 0:
    #             node.points += direction * step_size
    #
    #         node.agent_ids = original_ids
    #         print(f"🐦 CS 莱维飞行: Ball-{node.ball_id} 探索新区域")
    #     else:
    #         # --- 强制分裂 ---
    #         if node.split(self._get_next_id, self):  # ✅ 传入self作为env_instance
    #             # 分裂时，需要将原始节点的 agent_ids 分配给子节点
    #             # 获取分裂后的标签
    #             points = node.points
    #             if len(points) >= 2:
    #                 kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    #                 labels = kmeans.fit_predict(points)
    #
    #                 # 为每个子节点分配 agent_ids
    #                 for i, child in enumerate(node.children):
    #                     child_points = points[labels == i]
    #                     # 找到属于这个子节点的 agent_ids
    #                     child_agent_ids = []
    #                     for j, point in enumerate(points):
    #                         if labels[j] == i:
    #                             if j < len(original_ids):
    #                                 child_agent_ids.append(original_ids[j])
    #                     child.agent_ids = child_agent_ids
    #                     self.all_balls[child.ball_id] = child
    #
    #                 print(f"🐦 CS 寄生分裂: Ball-{node.ball_id} 分裂为两个子球")
    #         else:
    #             # 如果没有分裂，恢复 agent_ids
    #             node.agent_ids = original_ids

    # ==================== 核心算法：自适应重组 (融合 SFLA + CS) ====================
    def _adaptive_recluster(self):
        """
        混合策略主流程：
        1. SFLA: 所有叶子节点内部进行微调。
        2. CS: 质量差的叶子节点进行全局重组。
        """
        leaf_nodes = self._get_all_leaves(self.ball_tree)

        # 清理空节点
        valid_leaf_nodes = [node for node in leaf_nodes if len(node.points) > 0]
        if not valid_leaf_nodes:
            print("⚠️ 警告: 没有有效叶子节点，重建树结构")
            self._rebuild_tree_from_agents()
            leaf_nodes = self._get_all_leaves(self.ball_tree)
            valid_leaf_nodes = leaf_nodes

        # --- 阶段 1: SFLA 局部开发 ---
        for node in valid_leaf_nodes:
            if len(node.points) > 0:
                self._sfla_local_search(node)

        # --- 阶段 2: 质量评估与 CS 全局探索 ---
        split_candidates = []
        for node in valid_leaf_nodes:
            # 重新计算质量 (因为SFLA可能改变了位置)
            node.quality = self._calculate_cluster_quality(node)
            if node.quality < QTH and len(node.points) >= 4:
                split_candidates.append(node)

        # 执行 CS 重组
        for node in split_candidates:
            self._cs_global_restructure(node)

        # 重新分配智能体到更新后的树结构
        self._assign_agents_to_leaves()

        # ✅ 新增：更新所有非叶节点的子节点质量
        self._update_parent_nodes_quality(self.ball_tree)

    def _update_parent_nodes_quality(self, node):
        """递归更新所有非叶节点的子节点质量"""
        if node is None:
            return

        if not node.is_leaf and node.children:
            # 计算子节点的质量
            node.children_quality = []
            for child in node.children:
                child.quality = self._calculate_cluster_quality(child)
                node.children_quality.append(child.quality)

        # 递归处理子节点
        for child in node.children:
            self._update_parent_nodes_quality(child)

    # ==================== 辅助函数 ====================
    def _assign_agents_to_leaves(self):
        """分配智能体到叶子节点"""
        if self.ball_tree is None:
            print("⚠️ Warning: ball_tree is None")
            return

        leaf_nodes = self._get_all_leaves(self.ball_tree)
        if not leaf_nodes or len(self.env.agents) == 0:
            print(f"⚠️ Warning: No leaf nodes ({len(leaf_nodes)}) or no agents ({len(self.env.agents)})")
            return

        # print(f"🔍 Assigning {len(self.env.agents)} agents to {len(leaf_nodes)} leaf nodes")

        # 清空所有叶子节点的points和agent_ids
        for node in leaf_nodes:
            node.points = []
            node.agent_ids = []

        # 分配智能体
        for agent in self.env.agents:
            agent_pos = agent[:3]
            agent_id = int(agent[7])

            # 找到最近的叶子节点
            min_dist = float('inf')
            best_node = None
            for node in leaf_nodes:
                dist = np.linalg.norm(agent_pos - node.center)
                if dist < min_dist:
                    min_dist = dist
                    best_node = node

            if best_node is not None:
                best_node.points.append(agent_pos)
                best_node.agent_ids.append(agent_id)
            else:
                print(f"⚠️ Warning: No leaf node found for agent {agent_id}")

        # 更新节点特征
        for node in leaf_nodes:
            if len(node.points) > 0:
                node.update_features()
                # print(f"   Node {node.ball_id}: {len(node.agent_ids)} agents, center={node.center}")

    def _cleanup_empty_nodes(self, valid_leaf_nodes):
        """清理树中的空节点"""
        # 获取所有节点
        all_nodes = self._get_all_nodes(self.ball_tree)

        # 标记要删除的空节点
        nodes_to_remove = []
        for node in all_nodes:
            if hasattr(node, 'is_empty') and node.is_empty:
                nodes_to_remove.append(node)

        # 从树中移除空节点
        for node in nodes_to_remove:
            if node.parent:
                # 从父节点的子节点列表中移除
                if node in node.parent.children:
                    node.parent.children.remove(node)
                # 如果父节点现在只有一个子节点，将父节点替换为子节点
                if len(node.parent.children) == 1:
                    child = node.parent.children[0]
                    child.parent = node.parent.parent
                    if node.parent.parent:
                        # 更新祖父节点的子节点列表
                        for i, grandchild in enumerate(node.parent.parent.children):
                            if grandchild == node.parent:
                                node.parent.parent.children[i] = child
                                break
                    else:
                        # 如果父节点是根节点，更新根节点
                        self.ball_tree = child
            elif node == self.ball_tree:
                # 如果根节点是空的，用第一个有效节点替换
                if valid_leaf_nodes:
                    self.ball_tree = valid_leaf_nodes[0]
                    self.ball_tree.parent = None

        # 重新分配智能体到新的树结构
        # self._assign_agents_to_valid_leaves(valid_leaf_nodes)

    def _assign_agents_to_leaves1(self):
        leaf_nodes = self._get_all_leaves(self.ball_tree)
        if not leaf_nodes or len(self.agents) == 0:
            return

        # 清空所有叶子节点的 points 和 agent_ids
        for node in leaf_nodes:
            node.points = []
            node.agent_ids = []

        # 遍历每个智能体，分配到最近的叶子节点
        for agent in self._get_active_agents():
            agent_pos = agent[:3]
            agent_id = int(agent[7])
            best_node = min(leaf_nodes, key=lambda x: np.linalg.norm(agent_pos - x.center))
            best_node.points.append(agent_pos)
            best_node.agent_ids.append(agent_id)

        # 更新节点几何属性
        for node in leaf_nodes:
            if len(node.points) > 0:
                node.points = np.array(node.points)
                node.center = np.mean(node.points, axis=0)
                node.radius = np.max(cdist([node.center], node.points)) if len(node.points) > 1 else 0
            else:
                node.is_empty = True

        # 清理空节点（但不重新分配）
        self._cleanup_empty_nodes(leaf_nodes)

    def _rebuild_tree_from_agents(self):
        """当所有节点都为空时，从未承担守卫任务的智能体重建树。"""
        active_agents = self._get_active_agents()
        all_points = np.asarray([agent[:3] for agent in active_agents], dtype=float)
        self.ball_tree = GranularBallNode(self._get_next_id(), all_points)
        self.all_balls = {self.ball_tree.ball_id: self.ball_tree}

        self.ball_tree.agent_ids = [int(agent[7]) for agent in active_agents]

    def _get_all_nodes(self, node):
        """获取树中的所有节点"""
        if node is None:
            return []
        nodes = [node]
        for child in node.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes



    def _get_all_leaves(self, node):
        if node is None:
            return []
        if node.is_leaf:
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self._get_all_leaves(child))
        return leaves

    def _get_next_id(self):
        id_val = self.next_ball_id
        self.next_ball_id += 1
        return id_val

    def _calculate_cluster_quality(self, node):
        points = node.points
        n = len(points)
        if n == 0:
            return 0
        elif n == 1:
            return 0.5

        center = node.center
        distances = cdist(points, [center])
        avg_distance = np.mean(distances)
        internal_tightness = 1 / (1 + avg_distance / (ARENA_SIZE_XY * 0.1))

        ideal_min = MIN_CAPTURE_AGENTS
        ideal_max = MIN_CAPTURE_AGENTS + 3
        if n < ideal_min:
            scale_factor = n / ideal_min
        elif n <= ideal_max:
            scale_factor = 1.0
        else:
            scale_factor = max(0.5, 1.0 - (n - ideal_max) * 0.05)

        quality = (0.6 * internal_tightness + 0.4 * scale_factor)
        return quality

    def _calculate_fixed_circle_position(self, target_pos, agent_id, guard_ids, agent_type):
        """
        计算智能体在固定圆圈上的位置
        :param target_pos: 目标位置
        :param agent_id: 智能体ID
        :param guard_ids: 守卫智能体ID列表
        :param agent_type: 智能体类型 (0: UAV, 1: USV)
        :return: 期望位置
        """
        if agent_id not in guard_ids:
            return None

        idx = guard_ids.index(agent_id)
        n = len(guard_ids)

        # 圆圈半径 = CAPTURE_RADIUS/4 (因为直径是CAPTURE_RADIUS/2)
        circle_radius = CAPTURE_RADIUS / 4

        # 计算每个智能体的角度
        angle = 2 * np.pi * idx / n

        # 计算圆圈上的位置
        desired_x = target_pos[0] + circle_radius * np.cos(angle)
        desired_y = target_pos[1] + circle_radius * np.sin(angle)

        # 根据智能体类型设置高度。UAV 的三维守卫位置必须仍处于捕获半径内。
        if agent_type == 0:  # UAV
            max_vertical_offset = np.sqrt(max(CAPTURE_RADIUS ** 2 - circle_radius ** 2, 0.0))
            desired_z = target_pos[2] + min(200.0, 0.9 * max_vertical_offset)
        else:  # USV
            desired_z = target_pos[2]  # USV和目标同高

        return np.array([desired_x, desired_y, desired_z])

    def _maximum_angular_gap(self, agent_indices, target_pos):
        """计算目标周围候选智能体在 XY 平面上的最大方位角空缺。"""
        indices = np.asarray(agent_indices, dtype=int)
        if len(indices) < 2:
            return float(2 * np.pi)
        vectors = self.agents[indices, :2] - np.asarray(target_pos[:2], dtype=float)
        valid = np.linalg.norm(vectors, axis=1) > 1e-9
        vectors = vectors[valid]
        if len(vectors) < 2:
            return float(2 * np.pi)
        angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2 * np.pi)
        angles.sort()
        gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
        return float(np.max(gaps))

    def _capture_condition_met(self, in_range_indices, target_pos):
        """统一围捕判定：支持快速接近式和真正几何围捕两种模式。"""
        indices = np.asarray(in_range_indices, dtype=int)
        if len(indices) < MIN_CAPTURE_AGENTS:
            return False
        if CAPTURE_MODE == 'radius':
            return True
        if CAPTURE_MODE == 'geometric':
            gap = self._maximum_angular_gap(indices, target_pos)
            return bool(gap <= MAX_CAPTURE_ANGULAR_GAP + 1e-12)
        raise ValueError("CAPTURE_MODE must be 'radius' or 'geometric'")

    def _select_guard_indices(self, in_range_indices, target_pos, dists):
        """选择满足当前围捕定义的稳定守卫集合。"""
        candidates = [int(index) for index in np.asarray(in_range_indices, dtype=int)]
        candidates.sort(key=lambda index: float(dists[index]))
        if CAPTURE_MODE == 'radius':
            return candidates[:MIN_CAPTURE_AGENTS]

        selected = []
        remaining = candidates.copy()
        while remaining:
            if not selected:
                chosen = remaining.pop(0)
            else:
                selected_angles = np.mod(
                    np.arctan2(
                        self.agents[selected, 1] - target_pos[1],
                        self.agents[selected, 0] - target_pos[0],
                    ),
                    2 * np.pi,
                )
                def angular_score(index):
                    angle = float(np.mod(np.arctan2(
                        self.agents[index, 1] - target_pos[1],
                        self.agents[index, 0] - target_pos[0],
                    ), 2 * np.pi))
                    separations = np.abs(np.angle(np.exp(1j * (angle - selected_angles))))
                    return (float(np.min(separations)), -float(dists[index]))
                chosen = max(remaining, key=angular_score)
                remaining.remove(chosen)
            selected.append(chosen)
            if len(selected) >= MIN_CAPTURE_AGENTS and self._capture_condition_met(selected, target_pos):
                return selected

        # 完整候选集已经通过判定时，循环必然在此之前返回；保底使用最近成员。
        return candidates[:MIN_CAPTURE_AGENTS]

    def _release_redundant_captors(self, target_id, captor_ids, guard_ids):
        """Release non-guard captors and remove stale target-specific state immediately."""
        target_id = int(target_id)
        released = {int(agent_id) for agent_id in captor_ids} - {int(agent_id) for agent_id in guard_ids}
        if not released:
            return set()
        self.released_agents_last_step.update(released)

        self.previous_assignments = {
            int(agent_id): int(assigned_target)
            for agent_id, assigned_target in self.previous_assignments.items()
            if not (int(agent_id) in released and int(assigned_target) == target_id)
        }
        algorithm = getattr(self, 'algorithm', None)
        if algorithm is not None:
            for mapping_name in ('last_assignments', 'desired_waypoints', 'previous_slot_waypoints'):
                mapping = getattr(algorithm, mapping_name, None)
                if isinstance(mapping, dict):
                    for agent_id in released:
                        mapping.pop(int(agent_id), None)
            previous_plans = getattr(algorithm, 'previous_sfla_plans', None)
            if isinstance(previous_plans, dict):
                previous_plans.pop(target_id, None)
            active_targets = getattr(algorithm, 'sfla_active_targets', None)
            if isinstance(active_targets, set):
                active_targets.discard(target_id)
            if hasattr(algorithm, 'metrics'):
                algorithm.metrics['released_redundant_agents'] = (
                    int(algorithm.metrics.get('released_redundant_agents', 0)) + len(released)
                )
            if hasattr(algorithm, 'cs_stagnation_steps'):
                algorithm.cs_stagnation_steps = max(
                    int(getattr(algorithm, 'cs_stagnation_steps', 0)),
                    int(CS_STAGNATION_LIMIT),
                )
        return released

    def _update_capture_state(self):
        """Detect capture after movement, lock only guards, and release all redundancy."""
        current_locked = set()
        self.target_captors = {}
        self.released_agents_last_step = set()

        for target_id in range(len(self.targets)):
            target_pos = self.targets[target_id, :3]
            dists = np.linalg.norm(self.agents[:, :3] - target_pos, axis=1)
            in_range_indices = np.where(dists <= CAPTURE_RADIUS)[0]
            captor_ids = {int(self.agents[index, 7]) for index in in_range_indices}
            self.target_captors[target_id] = captor_ids

            if self._capture_condition_met(in_range_indices, target_pos):
                current_locked.add(target_id)
                if target_id not in self.guarding_agents:
                    guard_indices = self._select_guard_indices(in_range_indices, target_pos, dists)
                    guard_ids = {int(self.agents[index, 7]) for index in guard_indices}
                    self.guarding_agents[target_id] = guard_ids
                    print(f"\n🎯 目标 {target_id} 已被围捕成功！模式={CAPTURE_MODE}")
                    print(f"   参与围捕的智能体ID: {sorted(captor_ids)}")
                    print(f"   保留守卫智能体ID: {sorted(guard_ids)}")
                guard_ids = set(self.guarding_agents[target_id])
                released = self._release_redundant_captors(
                    target_id, captor_ids, guard_ids
                )
                if released:
                    print(f"   立即释放冗余智能体ID: {sorted(released)}")

        self.permanently_captured.update(current_locked)

    def _resolve_algorithm_waypoint(self, agent_id, target_id, fallback_position):
        """按算法是否提供有效航点决定运动，不再依赖算法名称硬编码。"""
        fallback = np.asarray(fallback_position, dtype=float)
        waypoints = getattr(self.algorithm, "desired_waypoints", {}) or {}
        waypoint = waypoints.get(int(agent_id))
        if waypoint is None:
            return fallback
        waypoint = np.asarray(waypoint, dtype=float)
        if waypoint.shape != (3,) or not np.all(np.isfinite(waypoint)):
            return fallback
        if int(target_id) in getattr(self, "permanently_captured", set()):
            return fallback
        return waypoint

    # ==================== 主循环逻辑 ====================
    def step(self):
        total_energy = 0
        # 1. 使用算法进行分组和任务分配。围捕判定在本步运动完成后执行。
        assignments = self.algorithm.step()  # 现在是 {智能体ID: 目标ID}
        assignments = self._sanitize_assignments(assignments)
        self._record_assignment_metrics(assignments)
        # 2. 获取分组节点（用于可视化同步）
        leaf_nodes = []
        if hasattr(self.algorithm, 'clusters_for_drawing'):
            leaf_nodes = self.algorithm.clusters_for_drawing
        elif hasattr(self, 'ball_tree') and self.ball_tree is not None:
            leaf_nodes = self._get_all_leaves(self.ball_tree)

        # 3. 执行移动
        for i in range(len(self.agents)):
            agent = self.agents[i]
            agent_pos = agent[:3]
            agent_id = int(agent[7])
            agent_type = int(agent[6])
            max_v = V_MAX_UAV if agent_type == 0 else V_MAX_USV

            # 检查是否为守卫智能体
            is_guarding = False
            guard_target_id = None
            for target_id, guards in self.guarding_agents.items():
                if agent_id in guards:
                    is_guarding = True
                    guard_target_id = target_id
                    break

            if is_guarding:
                # 执行守卫动作
                target_pos = self.targets[guard_target_id, :3]
                guard_ids = list(self.guarding_agents[guard_target_id])
                desired_pos = self._calculate_fixed_circle_position(target_pos, agent_id, guard_ids, agent_type)
                if desired_pos is None:
                    desired_pos = agent_pos
            else:
                # 非守卫智能体，根据算法分配移动
                if agent_id in assignments:
                    target_id = assignments[agent_id]
                    target_pos = self.targets[target_id, :3]
                    desired_pos = self._resolve_algorithm_waypoint(
                        agent_id, target_id, target_pos
                    )
                else:
                    # 没有分配到目标，保持原地
                    desired_pos = agent_pos

            # 动力学更新
            dx = desired_pos[0] - agent[0]
            dy = desired_pos[1] - agent[1]
            dz = desired_pos[2] - agent[2]
            distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            if distance < 1e-5: continue

            desired_theta = np.arctan2(dy, dx)
            desired_phi = np.arctan2(dz, np.sqrt(dx ** 2 + dy ** 2))
            new_v = min(distance, max_v)

            new_x = agent[0] + new_v * np.cos(desired_phi) * np.cos(desired_theta)
            new_y = agent[1] + new_v * np.cos(desired_phi) * np.sin(desired_theta)
            new_z = agent[2] + new_v * np.sin(desired_phi)
            if agent_type == 1: new_z = USV_Z

            self.agents[i, 0] = new_x
            self.agents[i, 1] = new_y
            self.agents[i, 2] = new_z
            self.agents[i, 3] = new_v
            self.agents[i, 4] = desired_theta
            self.agents[i, 5] = desired_phi
            self.total_travel_distance += new_v
            total_energy += new_v ** 2

        if TARGET_IS_STATIC == 0:
            self._update_targets()

        # 4. 在本步智能体/目标运动完成后立即判定围捕，避免捕获时间滞后一拍。
        self._update_capture_state()
        self.time_step += 1
        self.algorithm.metrics["total_distance"] = self.total_travel_distance
        self.algorithm.metrics["energy_consumption"] += total_energy

        # 同步粒球树（仅GB-SFLA需要）
        if self.algorithm_name in ["GB-SFLA", "GB-CS-SFLA", "GB-CS", "GB-only"]:
            self._sync_tree_with_agents()

        self.current_clusters_for_drawing = leaf_nodes

        return total_energy

    def _update_targets(self):
        """目标逃跑逻辑；随机种子仅在 reset() 时设置，随后连续推进随机流。"""
        for j in range(len(self.targets)):
            if j in self.permanently_captured:
                self.targets[j, 3] = 0
                continue
            target = self.targets[j]
            tx, ty, tz = target[0], target[1], TARGET_Z
            nearby_agents = []
            for agent in self.agents:
                dist_xy = np.sqrt((agent[0] - tx) ** 2 + (agent[1] - ty) ** 2)
                if dist_xy < TARGET_RUN_NUM:
                    nearby_agents.append((agent[0], agent[1], dist_xy))
            if nearby_agents:
                avg_x = sum([a[0] for a in nearby_agents]) / len(nearby_agents)
                avg_y = sum([a[1] for a in nearby_agents]) / len(nearby_agents)
                escape_dx = tx - avg_x
                escape_dy = ty - avg_y
                norm = np.sqrt(escape_dx ** 2 + escape_dy ** 2)
                if norm > 1e-5:
                    escape_dx = escape_dx / norm * TARGET_SPEED
                    escape_dy = escape_dy / norm * TARGET_SPEED
                else:
                    escape_dx, escape_dy = 0, 0
                target[0] += escape_dx
                target[1] += escape_dy
                target[3] = np.sqrt(escape_dx ** 2 + escape_dy ** 2)
                target[4] = np.arctan2(escape_dy, escape_dx)
            else:
                target[3] = TARGET_SPEED
                target[4] += random.uniform(-0.1, 0.1)
                target[0] += target[3] * np.cos(target[4])
                target[1] += target[3] * np.sin(target[4])
                target[2] = TARGET_Z

        # ==================== 新增：SFLA 与 CS 核心算法模块 ====================

    def _sync_tree_geometry_only(self):
        """只同步几何属性，不重新分配智能体ID"""
        leaf_nodes = self._get_all_leaves(self.ball_tree)
        if not leaf_nodes:
            return

        # 只更新几何属性，不清空agent_ids
        for node in leaf_nodes:
            if len(node.points) > 0:
                node.points = np.array(node.points)
                node.center = np.mean(node.points, axis=0)
                distances = cdist([node.center], node.points)
                node.radius = np.max(distances) if len(distances) > 0 else 0
            else:
                node.points = np.empty((0, 3))
                node.radius = 1.0

    def _assign_and_sync_agents(self):
        """统一的分配和同步函数"""
        leaf_nodes = self._get_all_leaves(self.ball_tree)
        if not leaf_nodes or len(self.agents) == 0:
            return

        # 清空所有叶子节点的 points 和 agent_ids
        for node in leaf_nodes:
            node.points = []
            node.agent_ids = []

        # 遍历每个智能体，分配到最近的叶子节点
        for agent in self._get_active_agents():
            agent_pos = agent[:3]
            agent_id = int(agent[7])
            best_node = min(leaf_nodes, key=lambda x: np.linalg.norm(agent_pos - x.center))
            best_node.points.append(agent_pos)
            best_node.agent_ids.append(agent_id)

        # 更新节点几何属性
        for node in leaf_nodes:
            if node.points:
                node.points = np.array(node.points)
                node.center = np.mean(node.points, axis=0)
                distances = cdist([node.center], node.points)
                node.radius = np.max(distances) if len(distances) > 0 else 0
            else:
                node.is_empty = True

        # 清理空节点
        self._cleanup_empty_nodes(leaf_nodes)

    def _sync_tree_with_agents(self):
        leaf_nodes = self._get_all_leaves(self.ball_tree)
        if not leaf_nodes:
            return

        # 1. 同时清空 points 和 agent_ids
        for node in leaf_nodes:
            node.points = []
            node.agent_ids = []  # 关键修复：清空 agent_ids

        # 2. 遍历所有智能体，同时分配坐标和ID
        for agent in self._get_active_agents():
            agent_pos = agent[:3]
            agent_id = int(agent[7])  # 提取智能体ID
            best_node = None
            best_dist = float('inf')
            for node in leaf_nodes:
                dist = np.linalg.norm(agent_pos - node.center)
                if dist < best_dist:
                    best_dist = dist
                    best_node = node
            if best_node is not None:
                best_node.points.append(agent_pos)
                # ✅ 添加去重检查：如果这个ID已经在这个节点中，就不要再添加
                if agent_id not in best_node.agent_ids:
                    best_node.agent_ids.append(agent_id)

        # 3. 更新几何属性
        for node in leaf_nodes:
            if node.points:
                node.points = np.array(node.points)
                node.center = np.mean(node.points, axis=0)
                distances = cdist([node.center], node.points)
                node.radius = np.max(distances) if len(distances) > 0 else 0
            else:
                node.points = np.empty((0, 3))
                node.radius = 1.0

    def print_final_tree_structure(self):
        """
        仿真结束时，打印完整的粒球树结构和包含的智能体ID
        修复：直接使用节点存储的agent_ids，不再依赖坐标匹配
        """
        print("\n" + "=" * 60)
        print("📊 仿真结束：最终粒球树结构与成员报告")
        print("=" * 60)

        # ✅ 新增：输出围捕总结
        print(f"\n🏆 围捕总结:")
        print(f"   总目标数: {TARGET_COUNT}")
        print(f"   成功围捕目标数: {len(self.permanently_captured)}")
        print(f"   剩余未围捕目标数: {TARGET_COUNT - len(self.permanently_captured)}")

        if self.guarding_agents:
            print(f"\n🛡️ 守卫智能体分配:")
            for target_id, guard_ids in self.guarding_agents.items():
                print(f"   目标 {target_id}: 守卫智能体 {sorted(list(guard_ids))}")

        print("\n" + "-" * 60)

        def _recursive_print(node, depth=0):
            indent = "  " * depth
            ball_id = node.ball_id
            point_count = len(node.points)
            quality = getattr(node, 'quality', 'N/A')
            node_type = "叶子节点(Leaf)" if node.is_leaf else "非叶节点(Internal)"

            # ✅ 显示子节点质量
            children_quality_str = ""
            if not node.is_leaf and hasattr(node, 'children_quality') and node.children_quality:
                children_quality_str = f", 子节点质量: {[round(q, 3) for q in node.children_quality]}"

            print(
                f"{indent}📂 Ball-{ball_id} | {node_type} | 数量: {point_count} | 质量: {quality:.3f}{children_quality_str}")

            # 直接使用存储的agent_ids，无需坐标匹配
            if node.is_leaf and hasattr(node, 'agent_ids') and node.agent_ids:
                id_str = ", ".join(map(str, sorted(node.agent_ids)))
                print(f"{indent}   🆔 包含智能体ID: [{id_str}]")
            elif node.is_leaf:
                print(f"{indent}   🆔 包含智能体ID: []")

            for child in node.children:
                _recursive_print(child, depth + 1)

        _recursive_print(self.ball_tree)


# ==================== 7. 绘图与主函数 (最终整合版) ====================





# ==================== 修改后的主函数 ====================
def main2():
    """运行多个算法并可视化对比"""
    # algorithms = ["GB-CS-SFLA", "GB-only"]
    # algorithms = ["GB-CS-SFLA", "KMeans-grouping"]
    # algorithms = ["GB-CS-SFLA", "GB-SFLA", "GB-CS", "GB-only", "KMeans-grouping", "CS-only","Standard-SFLA"]
    algorithms = ["GB-CS-SFLA","KMeans-grouping", "CS-only", "Standard-SFLA", "PSO", "GA"]
    # algorithms = ["GB-CS-SFLA", "GB-SFLA", "GB-CS", "GB-only"]
    # algorithms = ["GB-CS-SFLA", "GB-SFLA", "Standard-SFLA", "PSO", "GA"]
    # algorithms = ["GB-SFLA", "Standard-SFLA", "PSO", "GA"]
    results = {}

    if DIS_FIG:
        # 创建图形窗口用于对比显示
        plt.ion()
        n_cols = len(algorithms) // 2
        n_rows = int(np.ceil(len(algorithms) / n_cols))
        fig_compare = plt.figure(figsize=(5 * n_cols, 6 * n_rows))

        # 为每个算法创建子图
        axes = []
        for i, algo in enumerate(algorithms):
            ax = fig_compare.add_subplot(n_rows, n_cols, i + 1, projection='3d')
            axes.append(ax)


    # 运行每个算法
    for idx, algo_name in enumerate(algorithms):
        print(f"\n{'=' * 60}")
        print(f"Running {algo_name}...")
        print(f"{'=' * 60}")

        # 初始化环境
        env = SwarmEnv3D()
        env.set_algorithm(algo_name)

        # 获取当前子图
        if DIS_FIG:
            ax = axes[idx]
            ax.clear()

            # 设置子图标题
            ax.set_title(f"{algo_name}\nStep: 0", fontsize=10)
            ax.set_xlim(0, ARENA_SIZE_XY)
            ax.set_ylim(0, ARENA_SIZE_XY)
            ax.set_zlim(0, ARENA_SIZE_Z)

        # 运行仿真
        max_steps = 100
        capture_history = []

        for step in range(max_steps):
            # 执行一步
            total_energy = env.step()

            # # 每10步更新一次可视化
            if DIS_FIG:
                if step % 1 == 0:
                    print(f"  Step {step}: {len(env.permanently_captured)}/{len(env.targets)} targets captured")

                    # 清除子图
                    ax.clear()

                    # 绘制粒球
                    if hasattr(env, 'current_clusters_for_drawing'):
                        for node in env.current_clusters_for_drawing:
                            if hasattr(node, 'center') and hasattr(node, 'radius'):
                                center = node.center
                                radius = node.radius

                                # 绘制球体
                                u = np.linspace(0, 2 * np.pi, 10)
                                v = np.linspace(0, np.pi, 10)
                                x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                                y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                                z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                                ax.plot_surface(x, y, z, alpha=0.1, color='blue')

                    # 绘制智能体
                    for agent in env.agents:
                        x, y, z = agent[:3]
                        agent_type = int(agent[6])
                        color = 'cyan' if agent_type == 0 else 'limegreen'
                        marker = '^' if agent_type == 0 else 's'
                        ax.scatter(x, y, z, c=color, marker=marker, s=10)

                    # 绘制目标
                    for i, target in enumerate(env.targets):
                        tx, ty, tz = target[:3]
                        color = 'green' if i in env.permanently_captured else 'red'
                        ax.scatter(tx, ty, tz, c=color, s=10, marker='o')

                        # 绘制捕获范围
                        if i not in env.permanently_captured:
                            u = np.linspace(0, 2 * np.pi, 20)
                            v = np.linspace(0, np.pi, 20)
                            x = tx + CAPTURE_RADIUS * np.outer(np.cos(u), np.sin(v))
                            y = ty + CAPTURE_RADIUS * np.outer(np.sin(u), np.sin(v))
                            z = tz + CAPTURE_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))
                            ax.plot_surface(x, y, z, alpha=0.1, color=color)

                    # 更新标题
                    ax.set_title(
                        f"{algo_name}\nStep: {step}, Captured: {len(env.permanently_captured)}/{len(env.targets)}",
                        fontsize=10)
                    ax.set_xlim(0, ARENA_SIZE_XY)
                    ax.set_ylim(0, ARENA_SIZE_XY)
                    ax.set_zlim(0, ARENA_SIZE_Z)

                    # 刷新图形
                    fig_compare.canvas.draw()
                    fig_compare.canvas.flush_events()
                    plt.pause(0.01)

            # 记录捕获历史
            capture_history.append(len(env.permanently_captured))

            # 检查是否完成
            if len(env.permanently_captured) >= len(env.targets):
                print(f"✅ {algo_name} completed in {step + 1} steps!")
                break

        # 记录结果
        results[algo_name] = {
            "capture_time": step + 1,
            "total_travel_distance": env.total_travel_distance,
            "target_load_balance": (
                float(np.mean(env.target_load_balance_history))
                if env.target_load_balance_history else 0.0
            ),
        }

        print(f"📊 Results for {algo_name}:")
        print(f"   Capture Time: {results[algo_name]['capture_time']} steps")
        print(f"   Total Travel Distance: {results[algo_name]['total_travel_distance']:.2f}")
        print(f"   Target Load Balance (mean CV): {results[algo_name]['target_load_balance']:.4f}")

    # print('----------------------------\n','results',results)


    # 打印表头
    print(f"{'Algorithm':<20} {'Capture Time':>14} {'Total Travel Distance':>22} {'Target Load Balance':>20}")
    print("-" * 80)

    # 遍历字典，提取并输出每个算法的指标
    for algo, metrics in results.items():
        print(
            f"{algo} \t{metrics['capture_time']} \t{metrics['total_travel_distance']:.2f} \t{metrics['target_load_balance']:.6f}")
    # 创建性能对比图表
    create_performance_comparison(results, algorithms)




def create_performance_comparison(results, algorithms):
    """创建性能对比图表"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    colors = ['green', 'blue', 'red', 'orange', 'purple', 'brown', 'cyan', 'magenta']


    def annotate_bars(axis, bars, formatter):
        for bar in bars:
            height = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                formatter(height),
                ha='center',
                va='bottom',
            )

    # 1. 捕获时间对比
    ax1 = axes[0]
    capture_times = [results[algo]["capture_time"] for algo in algorithms]
    bars1 = ax1.bar(algorithms, capture_times, color=colors[:len(algorithms)])
    ax1.set_title('Capture Time Comparison')
    ax1.set_ylabel('Steps')
    ax1.tick_params(axis='x', rotation=45)
    annotate_bars(ax1, bars1, lambda value: f'{int(value)}')

    # 2. 总航程对比
    ax4 = axes[1]
    total_travel_distances = [results[algo]["total_travel_distance"] for algo in algorithms]
    bars4 = ax4.bar(algorithms, total_travel_distances, color=colors[:len(algorithms)])
    ax4.set_title('Total Travel Distance Comparison')
    ax4.set_ylabel('Distance')
    ax4.tick_params(axis='x', rotation=45)
    annotate_bars(ax4, bars4, lambda value: f'{value:.0f}')

    # 3. 活跃目标负载均衡对比
    ax5 = axes[2]
    target_load_balances = [results[algo]["target_load_balance"] for algo in algorithms]
    bars5 = ax5.bar(algorithms, target_load_balances, color=colors[:len(algorithms)])
    ax5.set_title('Target Load Balance Comparison')
    ax5.set_ylabel('Mean Load CV (Lower is Better)')
    ax5.tick_params(axis='x', rotation=45)
    annotate_bars(ax5, bars5, lambda value: f'{value:.3f}')

    plt.tight_layout()

    from datetime import datetime
    now = datetime.now()
    month = now.month  # 月 (1-12)
    day = now.day  # 日 (1-31)
    hour = now.hour  # 小时 (0-23)
    minute = now.minute  # 分钟 (0-59)

    plt.savefig('./result_png/algorithm_comparison' +
                str(INIT_RANDOMSTATE) +
                str(UAV_COUNT) +
                str(USV_COUNT) +
                str(TARGET_COUNT) +
                str(month) +
                str(day) +
                str(hour) +
                str(minute) +
                '.png',
                dpi=150,
                bbox_inches='tight')
    print("\n📊 Performance comparison chart saved as 'algorithm_comparison  "+
          str(INIT_RANDOMSTATE) +
          str(UAV_COUNT) +
          str(USV_COUNT) +
          str(TARGET_COUNT)
          )
    plt.ioff()
    plt.show()

# ==================== 修改后的主函数入口 ====================
if __name__ == "__main__":

    print("🚀 Running algorithm comparison with visualization...")
    main2()

