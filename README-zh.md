# DiffusionX

[English](README.md) | 简体中文

> [!NOTE]
> 开发正在进行中。DiffusionX 是多线程高性能 Rust 随机数/随机过程模拟库的 Python 绑定，通过 [PyO3](https://github.com/PyO3/pyo3) 实现。

## 使用示例

```python
from diffusionx.simulation import Bm

# 布朗运动模拟
bm = Bm() 
traj = bm(10)
times, positions = traj.simulate(step_size=0.01)  # 模拟布朗运动轨迹，返回 ndarray 数组

# 蒙特卡罗模拟布朗运动的统计量
raw_moment = traj.raw_moment(order=1, particles=1000)  # 一阶原点矩
central_moment = traj.central_moment(order=2, particles=1000)  # 二阶中心矩

# 布朗运动首次通过时间
fpt = bm.fpt((-1, 1))
```

## 进展
### 随机数生成

- [x] 正态分布
- [x] 均匀分布
- [x] 指数分布
- [x] 泊松分布
- [x] alpha 稳定分布

### 随机过程

- [x] 布朗运动
- [x] alpha 稳定 Lévy 过程
- [x] 从属过程
- [x] 逆从属过程
- [x] 分数布朗运动
- [x] 泊松过程
- [ ] 复合泊松过程
- [x] Langevin 过程
- [x] 广义 Langevin 方程
- [x] 从属 Langevin 方程

### 泛函

- [x] 首次通过时间
- [x] 停留时间

## Benchmark

### 测试结果

生成长度为 `10_000_000` 的随机数组

|                          | 标准正态分布 | `[0, 1]` 均匀分布 |  稳定分布  |
| :----------------------: | :----------: | :---------------: | :--------: |
|  DiffusionX (Rust 版本)  |  17.576 ms   |     15.131 ms     | 133.85 ms  |
| DiffusionX (Python 版本) |   41.2 ms    |     34.3 ms     |  293 ms  |
|          Julia           |  27.671 ms   |     12.755 ms      | 570.260 ms |
|      NumPy / SciPy       |    199 ms    |      66.6 ms      |   1.67 s   |
|          Numba           |      -       |         -         |   1.15 s   |

### 测试环境

#### 硬件配置
- 设备型号：MacBook Air 13-inch (2024)
- 处理器：Apple M3 
- 内存：16GB

#### 软件环境
- 操作系统：macOS Sequoia 15.3
- Rust：1.85.0
- Python：3.12
- Julia：1.11
- NumPy：2
- SciPy：1.15.1

## 技术栈 & 特性

- 🦀 Rust 2024 Edition
- 🔄 PyO3：Rust/Python 绑定
- 🔢 NumPy：零开销数组转换
- 🚀 高性能 
- 🔄 零开销 NumPy 兼容：所有随机数生成函数直接返回 NumPy 数组

## 许可证

本项目采用双许可证模式：

* [MIT 许可证](https://opensource.org/licenses/MIT)
* [Apache 许可证 2.0 版本](https://www.apache.org/licenses/LICENSE-2.0)

您可以选择使用其中任一许可证。
