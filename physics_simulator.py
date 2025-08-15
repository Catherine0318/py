# physics_simulator.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
import streamlit as st


class ParticleSystem:
    def __init__(self, n=10000, simulation_mode='temperature', physical_param=1.0):
        """初始化粒子系统

        参数:
            n: 粒子数量
            simulation_mode: 模拟模式('temperature'或'mass')
            physical_param: 物理参数(温度或质量值)
        """
        self.n = n
        self.simulation_mode = simulation_mode
        self.physical_param = physical_param  # 重命名为physical_param避免冲突
        self.box_size = 15
        self.pos = np.random.rand(n, 2) * self.box_size
        self._initialize_velocities()

    def _initialize_velocities(self):
        """根据当前模式初始化粒子速度"""
        if self.simulation_mode == 'temperature':
            scale = self.physical_param  # 温度模式使用标准差
        else:  # mass模式
            scale = np.sqrt(1 / self.physical_param)

        speed = maxwell.rvs(scale=scale, size=self.n)
        angle = np.random.rand(self.n) * 2 * np.pi
        self.vel = np.column_stack([speed * np.cos(angle), speed * np.sin(angle)])

    def update(self):
        """更新粒子位置"""
        self.pos += self.vel * 0.03
        self._handle_boundary()

    def _handle_boundary(self):
        """处理边界碰撞"""
        self.vel[:, 0] = np.where(
            (self.pos[:, 0] < 0) | (self.pos[:, 0] > self.box_size),
            -self.vel[:, 0], self.vel[:, 0]
        )
        self.vel[:, 1] = np.where(
            (self.pos[:, 1] < 0) | (self.pos[:, 1] > self.box_size),
            -self.vel[:, 1], self.vel[:, 1]
        )

    @property
    def speeds(self):
        """计算所有粒子的速度大小"""
        return np.linalg.norm(self.vel, axis=1)


# Streamlit界面配置
st.set_page_config(layout="wide")
st.title("🌡️ 粒子物理模拟器")

# 侧边栏控制面板
with st.sidebar:
    st.header("控制参数")
    current_mode = st.radio("模拟模式", ['温度变化', '质量变化'], index=0)
    particle_count = st.slider("粒子数量", 1000, 100000, 10000, step=1000)

    # 根据模式显示不同参数控件
    if current_mode == '温度变化':
        temperature_value = st.slider("温度参数", 0.1, 20.0, 1.0, step=0.1)
        mode_param = 'temperature'
        physical_param_value = temperature_value
    else:
        mass_value = st.slider("粒子质量", 0.1, 10.0, 1.0, step=0.1)
        mode_param = 'mass'
        physical_param_value = mass_value

# 初始化系统
system = ParticleSystem(
    n=particle_count,
    simulation_mode=mode_param,
    physical_param=physical_param_value
)

# 实时更新模拟
system.update()

# 创建可视化图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 粒子运动图
ax1.scatter(system.pos[:, 0], system.pos[:, 1], s=1, alpha=0.5, color='blue')
ax1.set_xlim(0, system.box_size)
ax1.set_ylim(0, system.box_size)
ax1.set_title("粒子运动轨迹", fontsize=12)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.grid(True, alpha=0.2)

# 速度分布图
max_speed = 5 if mode_param == 'mass' else min(20, max(5, 4 * physical_param_value))
speed_range = np.linspace(0, max_speed, 300)

if mode_param == 'temperature':
    theory_curve = maxwell.pdf(speed_range, scale=physical_param_value)
    title = f"速度分布 (温度={physical_param_value:.1f})"
else:
    theory_curve = maxwell.pdf(speed_range, scale=np.sqrt(1 / physical_param_value))
    title = f"速度分布 (质量={physical_param_value:.1f})"

ax2.plot(speed_range, theory_curve, 'r-', lw=2, label='理论值')
ax2.hist(system.speeds, bins=50, density=True, alpha=0.5, label='模拟值')
ax2.set_title(title, fontsize=12)
ax2.set_xlim(0, max_speed)
ax2.legend()
ax2.grid(True, alpha=0.2)

# 显示图表
st.pyplot(fig)

# 添加说明
with st.expander("ℹ️ 使用说明"):
    st.markdown("""
    - **温度模式**：调节粒子运动速度的分布宽度
    - **质量模式**：调节粒子质量（影响速度分布形状）
    - 粒子数量越多，模拟越精确但性能消耗越大
    """)