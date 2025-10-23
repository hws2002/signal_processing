<div style="display: flex; justify-content: space-between; font-size: 9px; margin-bottom: 10px;">
  <span>2021080070 计31 韩佑硕</span>
  <span><script>document.write(new Date().toISOString().split('T')[0])</script></span>
</div>

# 实验一 ： 傅里叶级数的可视化

# 代码分析

## `visualize()`

### 전체 구조

Fourier 급수를 회전하는 원들의 연쇄로 시각화합니다. 시간에 따라 원들이 회전하면서 마지막 원의 끝점이 원본 함수를 추적합니다.


**1. 원본 함수**
```python
  x = np.linspace(0, 2π, 1000)   # 0~2π를 1000개 점으로 나눔
  y[i] = function(x[i])          # 각 점에서 f(t) 값
```
여기에 해당하는 `function(t)` 를 구현해야 함.

**2. 원 데이터 array**

```python
    point_pos_array = np.zeros((2*N_Fourier + 2, 2))  # 각 원의 끝점 위치 [x, y]
    radius_array = np.zeros(2*N_Fourier + 1)          # 각 원의 반지름
```

인덱스 구조:  

- [0]: 원점(0, 0)
- [1]: a0 원의 끝점
- [2]: b1 원의 끝점
- [3]: a1 원의 끝점
- [4]: b2 원의 끝점
- [5]: a2 원의 끝점

### 회전 벡터 
- 회전 벡터: radius × [cos(ωt), sin(ωt)]
- 각속도 ω = (j+1) → 1, 2, 3, ...

**3. b_n 원**     
- cos 항: b_n × sin(nt)의 x, y 성분으로 분해

**4. a_n 원**
- cos 항: a_n × cos(nt) = a_n × [-sin(nt), cos(nt)]
- b_n 원보다 90도 앞서 회전

**5. 그래프 그리기**

**6. 애니메이션 생성 (exp2.py:122-125)**
- 100개 프레임 → mp4 영상

결과: 원들이 회전하면서 반원파를 그려나가는 애니메이션!

**핵심 원리**

마지막 원의 끝점 = (x_끝, y_끝)

y_끝 = a0 + a1·cos(1·t) + b1·sin(1·t) + a2·cos(2·t) + b2·sin(2·t) + ...
    = Fourier 급수 근사값!

- y_끝 ≈ f(t): N_Fourier가 클수록 정확

## 自加函数 `compare_values`

为了量化评估傅里叶级数近似的精度，添加了实际值与预测值的对比分析功能。

该函数计算给定时间点 $t$ 处的傅里叶级数近似值：
$$
\hat{f}(t) = a_0 + \sum_{n=1}^{N} [a_n \cos(nt) + b_n \sin(nt)]
$$

### 2. 误差指标

在100个测试点上计算以下误差指标：

**均方误差 (MSE)**
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (f(t_i) - \hat{f}(t_i))^2
$$

**平均绝对误差 (MAE)**
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |f(t_i) - \hat{f}(t_i)|
$$

**最大误差**
$$
\text{Max Error} = \max_{i} |f(t_i) - \hat{f}(t_i)|
$$

### 3. 可视化输出

生成双子图对比分析：
- **左图**：蓝色实线（实际值）vs 红色虚线（预测值）
- **右图**：绿色线显示误差 $e(t) = f(t) - \hat{f}(t)$ 随时间的变化

### 4. 结果分析

**方波信号特点**：
- 不连续点处出现 Gibbs 现象
- $N \geq 32$ 时大部分区间近似良好
- 最大误差集中在不连续点附近

**半圆波信号特点**：
- 连续且光滑，收敛速度快
- $N \geq 64$ 时精度很高
- 端点处（导数无穷大）误差略大


# 任务一 ： 可视化方波信号

## 傅里叶系数

### 원함수 분석

주어진 함수 `f(t) = 0.5·sgn(sin(t)) + 0.5` 는 주기가 $T = 2\pi$ 이고, 다음과 같은 사각파(Square Wave) 형태를 가집니다.  

$$
f(t) = \begin{cases} 1 & \text{if } 0 < t < \pi \\ 0 & \text{if } \pi < t < 2\pi \end{cases}
$$

### 1. $a_0$ 계산

$$
\begin{aligned}
a_0 &= \frac{1}{2\pi} \int_{0}^{2\pi} f(t) \,dt \\
&= \frac{1}{2\pi} \left( \int_{0}^{\pi} 1 \,dt + \int_{\pi}^{2\pi} 0 \,dt \right) \\
&= \frac{1}{2\pi} \left( [t]_{0}^{\pi} \right) \\
&= \frac{1}{2\pi} (\pi) = \frac{1}{2}
\end{aligned}
$$

### 2. $a_n$ 계산 (코사인 계수)

$n \ge
 1$인 경우
$$
\begin{aligned}
a_n &= \frac{1}{\pi} \int_{0}^{2\pi} f(t) \cos(nt) \,dt \\
&= \frac{1}{\pi} \int_{0}^{\pi} 1 \cdot \cos(nt) \,dt \\
&= \frac{1}{\pi} \left[ \frac{\sin(nt)}{n} \right]_{0}^{\pi} \\
&= \frac{1}{n\pi} (\sin(n\pi) - \sin(0)) \\
&= 0
\end{aligned}
$$

따라서 모든 코사인 계수는 0


### 3. $b_n$ 계산 (사인 계수)
$$
\begin{aligned}
b_n &= \frac{1}{\pi} \int_{0}^{2\pi} f(t) \sin(nt) \,dt \\
&= \frac{1}{\pi} \int_{0}^{\pi} 1 \cdot \sin(nt) \,dt \\
&= \frac{1}{\pi} \left[ -\frac{\cos(nt)}{n} \right]_{0}^{\pi} \\
&= -\frac{1}{n\pi} (\cos(n\pi) - \cos(0)) \\
&= -\frac{1}{n\pi} ((-1)^n - 1)
\end{aligned}
$$  

즉,

$$
b_n =
\begin{cases}
\dfrac{2}{n\pi}, & \text{if } n \equiv 1 \pmod{2} \\[6pt]
0, & \text{if } n \equiv 0 \pmod{2}
\end{cases}
$$


### 최종 푸리에 급수

계산된 계수들을 종합하면 최종 푸리에 급수는 다음과 같습니다.
$$
f(t) = \frac{1}{2} + \sum_{n=1, 3, 5, \dots}^{\infty} \frac{2}{n\pi} \sin(nt)
$$

혹은 
$$
f(t) = \frac{1}{2} + \frac{2}{\pi} \sum_{k=1}^{\infty} \frac{\sin((2k-1)t)}{2k-1}
$$

## 可视化结果

* n이 비교적 작은 수일때는, 꽤나 큰 오차가 있었다.(n = 2,4)

* n=8부터 오차가 비교적 작아지기 시작하고

* n=32부터는 거의 같은 값이라고 봐도 무방할 정도로 red line이 평행을 유지한다.



# 任务二 ： 可视化半圆波信号

## 傅里叶系数

###  원함수 분석

주어진 반원파 신호(half-circle wave signal) 함수는 다음과 같습니다.
$$
f(t) = \begin{cases} \sqrt{\pi^2 - (t-\pi)^2} & \text{if } t \in [0, 2\pi] \\ f(t-2\pi) & \text{if } t \notin [0, 2\pi] \end{cases}
$$
이 함수는 중심이 $(\pi, 0)$이고 반지름이 $\pi$인 상반원이며, 주기가 $T=2\pi$입니다.

### 1. $a_0$ 계산 


$$
a_0 = \frac{1}{2\pi} \int_{0}^{2\pi} f(t) \,dt = \frac{1}{2\pi} \int_{0}^{2\pi} \sqrt{\pi^2 - (t-\pi)^2} \,dt
$$
위의 정적분은 반지름이 $\pi$인 반원의 넓이와 같으므로, 
$$
a_0 = \frac{1}{2\pi} \left( \frac{\pi^3}{2} \right) = \frac{\pi^2}{4}
$$

### 2. $b_n$

함수 $f(t)$는 $t=\pi$에 대해 좌우 대칭인 우함수(even function)의 특징을 가집니다. 이러한 대칭성으로 인해 사인 계수인 $b_n$은 모두 0이 됩니다. 이를 수식으로 증명하면 다음과 같습니다.
$$
b_n = \frac{1}{\pi} \int_{0}^{2\pi} f(t) \sin(nt) \,dt
$$
$u = t - \pi$로 치환하면, $t = u + \pi$이고 $dt = du$ 입니다.
$$
\begin{aligned}
b_n &= \frac{1}{\pi} \int_{-\pi}^{\pi} \sqrt{\pi^2 - u^2} \sin(n(u+\pi)) \,du \\
&= \frac{1}{\pi} \int_{-\pi}^{\pi} \sqrt{\pi^2 - u^2} \left( \sin(nu)\cos(n\pi) + \cos(nu)\sin(n\pi) \right) \,du \\
&= \frac{(-1)^n}{\pi} \int_{-\pi}^{\pi} \underbrace{\sqrt{\pi^2 - u^2}}_{\text{우함수}} \underbrace{\sin(nu)}_{\text{기함수}} \,du
\end{aligned}
$$
우함수와 기함수의 곱은 기함수이므로, 대칭 구간 $[-\pi, \pi]$에서의 정적분 값은 0입니다.
$$
b_n = 0
$$

### 3. $a_n$ 계산

$n \ge 1$인 경우,

$$
a_n = \frac{1}{\pi} \int_{0}^{2\pi} f(t) \cos(nt) \,dt
$$
$b_n$과 동일하게 $u = t - \pi$로 치환합니다.
$$
\begin{aligned}
a_n &= \frac{1}{\pi} \int_{-\pi}^{\pi} \sqrt{\pi^2 - u^2} \cos(n(u+\pi)) \,du \\
&= \frac{1}{\pi} \int_{-\pi}^{\pi} \sqrt{\pi^2 - u^2} \left( \cos(nu)\cos(n\pi) - \sin(nu)\sin(n\pi) \right) \,du \\
&= \frac{(-1)^n}{\pi} \int_{-\pi}^{\pi} \underbrace{\sqrt{\pi^2 - u^2}}_{\text{우함수}} \underbrace{\cos(nu)}_{\text{우함수}} \,du
\end{aligned}
$$
피적분함수가 우함수이므로, 적분 구간을 절반으로 줄이고 2를 곱할 수 있습니다.
$$
a_n = \frac{2(-1)^n}{\pi} \int_{0}^{\pi} \sqrt{\pi^2 - u^2} \cos(nu) \,du
$$
이 적분은 직접 계산하기 어려우므로, 수치해석적 방법을 써서 계산해냈다. 


## 数值积分
대표적인 机械法를 써서 구해냈다.

### Gauss-Legendre 积分法

**원리:**
- 구간 $[-1, 1]$에서 최적화된 节点 $x_i$와 权重 $w_i$를 사용
- 임의 구간 $[a, b]$로 변환: $t = \frac{b-a}{2}x_i + \frac{a+b}{2}$
- 적분 근사: $\int_a^b f(t)\,dt \approx \frac{b-a}{2} \sum_{i=1}^{n} w_i \cdot f(t_i)$

**구현:**
```python
GL_nodes, GL_weights = np.polynomial.legendre.leggauss(NUM_POINTS)

def gauss_legendre(f, n, a, b):
    res = 0.0
    for i in range(NUM_POINTS):
        t = 0.5*(b-a)*GL_nodes[i] + 0.5*(a+b)
        w = GL_weights[i]
        res += w * f(t, n)
    res *= 0.5*(b-a)
    return res
```

**특징:**
- 구현이 매우 간단 (节点과 权重만 있으면 됨)
- 부드러운 함수에 대해 높은 정확도
- 하지만 节点이 125개 정도 있어야 `N_Fourier=128`인 경우에도 안정적인 근사값을 구해낼 수 있었다
- 특히 **낮은 n값 (n=1,2)에서 불안정**: 끝점 $t=\pi$에서 $\sqrt{\pi^2-t^2} \to 0$이고 도함수가 무한대로 발산하여 GL 적분 오차 발생

**정확도:**
- `NUM_POINTS = 300`: 모든 N_Fourier에서 안정적
- `NUM_POINTS = 125`: N_Fourier=128까지 가능
- `NUM_POINTS = 50`: 낮은 차수에서도 오차 발생

## 可视化结果

### 实验数据汇总

对不同的傅里叶级数项数 $N$ 进行了近似精度测试，在100个等距采样点上计算误差指标。结果如下：

| $N$ | MSE | MAE | Max Error |
|-----|---------|---------|-----------|
| 2 | 0.057117 | 0.149177 | 1.239647 |
| 8 | 0.010920 | 0.041603 | 0.682521 |
| 16 | 0.005205 | 0.021305 | 0.491135 |
| 32 | 0.002579 | 0.012229 | 0.350399 |
| 64 | 0.001252 | 0.006805 | 0.248903 |
| 128 | 0.000625 | 0.004236 | 0.176437 |

### 误差分析

**收敛趋势：**
- 随着 $N$ 的增加，所有误差指标呈现指数级下降
- MSE 从 0.057 ($N=2$) 降至 0.0006 ($N=128$)，提升约 **90倍**
- MAE 从 0.149 ($N=2$) 降至 0.004 ($N=128$)，提升约 **35倍**
- Max Error 从 1.240 ($N=2$) 降至 0.176 ($N=128$)，提升约 **7倍**

**关键观察：**
1. **$N=2$**：近似效果较差，最大误差超过1.2，仅能捕捉函数的基本形状
2. **$N=8-16$**：误差快速下降，开始呈现半圆波的基本轮廓
3. **$N=32$**：MSE < 0.003，大部分区域近似良好
4. **$N≥64$**：MAE < 0.01，视觉上几乎完美重合
5. **$N=128$**：MSE仅为0.0006，达到极高精度

**误差分布特征：**
- 最大误差始终出现在 $t=0$ (和 $t=2\pi$) 处
- 这是因为半圆波在端点处导数趋于无穷大 ($\lim_{t\to 0^+} f'(t) = +\infty$)
- 中间区域 ($t \in [\pi/2, 3\pi/2]$) 收敛最快，误差最小
- Gibbs 现象不明显，因为函数本身连续且光滑（除端点外）

**性能提升规律：**

每当 $N$ 翻倍，MSE 大约减半：
- $N: 8 \to 16$, MSE: $0.0109 \to 0.0052$ (减少52%)
- $N: 16 \to 32$, MSE: $0.0052 \to 0.0026$ (减少50%)
- $N: 32 \to 64$, MSE: $0.0026 \to 0.0013$ (减少51%)
- $N: 64 \to 128$, MSE: $0.0013 \to 0.0006$ (减少50%)

这表明傅里叶级数对光滑周期函数具有良好的收敛性。

### 可视化观察

* **$N=2, 8$**：红色虚线（预测值）在端点附近明显偏离蓝色实线（实际值）
* **$N=16, 32$**：整体拟合良好，但在 $t \in [0, 1]$ 和 $t \in [5, 6]$ 区间仍有轻微波动
* **$N≥64$**：预测曲线几乎完全重合于实际曲线，仅在端点处有微小偏差

误差曲线（右图）显示：
- 端点处误差呈现尖峰
- 中间平滑区域误差接近0
- 随着 $N$ 增大，尖峰高度显著降低

## Others

lru_cache를 사용해 한번의 계산으로 coefficient를 저장해 다시 사용하게 했다.