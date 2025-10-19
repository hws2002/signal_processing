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
  x = np.linspace(0, 2π, 1000)  # 0~2π를 1000개 점
  y[i] = function(x[i])          # 각 점에서 f(t) 값
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

**3. b_n 원들**     
    - cos 항: b_n × sin(nt)의 x, y 성분으로 분해

**4. a_n 원들**
    - sin/cos 순서 바뀜 → 90도 위상차
    - sin 항: a_n × cos(nt)가 아닌 회전 벡터의 x, y 성분

**5. 그래프 그리기**

**6. 핵심 원리**

마지막 원의 끝점 = (x_끝, y_끝)

y_끝 = a0 + a1·cos(1·t) + b1·sin(1·t)
        + a2·cos(2·t) + b2·sin(2·t)
        + ...
    = Fourier 급수 근사값!

- y_끝 ≈ f(t): N_Fourier가 클수록 정확
- 빨간선 길이: 근사 오차
- 시간 진행: 끝점이 원본 함수 궤적을 따라감

  8. 애니메이션 생성 (exp2.py:122-125)

  100개 프레임 → mp4 영상

  결과: 원들이 회전하면서 반원파를 그려나가는 애니메이션!

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
대표적인 机械法들을 써서 구해냈다.

### 1. Gauss-Legendre 积分法

먼저 쓴 방법은 Gauss-Legendre방식이다.

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
- 특히 **낮은 n값 (n=1,2)에서 불안정**: 끝점 $t=\pi$에서 $\sqrt{\pi^2-t^2} \to 0$의 특이성 때문

**정확도:**
- `NUM_POINTS = 300`: 모든 N_Fourier에서 안정적
- `NUM_POINTS = 125`: N_Fourier=128까지 가능
- `NUM_POINTS = 50`: 낮은 차수에서도 오차 발생

### 2. FFT (Fast Fourier Transform)

더 빠르고 정확한 방법으로 **Cooley-Tukey Radix-2 DIT FFT 알고리즘**을 직접 구현하여 사용했다.

**원리:**
- 함수를 균일하게 샘플링: $t_k = \frac{2\pi k}{N}, k=0,1,...,N-1$ (N은 2의 거듭제곱)
- Cooley-Tukey FFT로 주파수 성분 추출 (DFT를 $O(N\log N)$으로 계산)
- Fourier 계수로 변환:
  - $a_0 = \frac{1}{N} \text{Re}[X_0]$
  - $a_m = \frac{2}{N} \text{Re}[X_m]$, $(m \geq 1)$
  - $b_m = -\frac{2}{N} \text{Im}[X_m]$

**구현 (Cooley-Tukey Radix-2 DIT):**
```python
def fft_manual(x):
    N = len(x)
    if N <= 1: return x

    # 1단계: 비트 역순 재정렬 (bit-reversal permutation)
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j: x[i], x[j] = x[j], x[i]

    # 2단계: 버터플라이 연산 (butterfly operations)
    block_size = 2
    while block_size <= N:
        half_block = block_size // 2
        angle = -2π / block_size
        twiddle_base = e^(i·angle)  # 회전인자
        for i in range(0, N, block_size):
            twiddle = 1.0
            for j in range(half_block):
                k = i + j
                temp = twiddle * x[k + half_block]
                x[k + half_block] = x[k] - temp
                x[k] = x[k] + temp
                twiddle *= twiddle_base
        block_size <<= 1
    return x
```

**알고리즘 설명:**
1. **비트 역순 재정렬**: 입력 데이터를 인덱스의 비트를 뒤집은 순서로 재배치
2. **버터플라이 연산**: 분할 정복 방식으로 DFT 계산
   - block_size를 2, 4, 8, ... N까지 2배씩 증가
   - 각 블록 내에서 회전인자(twiddle factor)를 곱해 합산

**Fourier 계수 계산:**
```python
def calculate_all_coeffs_manual(num_coeffs, N_samples=2048):
    t = np.linspace(0, 2π, N_samples, endpoint=False)
    y = semi_circle_wave(t).tolist()
    fft_result = fft_manual(y)

    a0 = fft_result[0].real / N_samples
    a_m = 2 * [fft_result[m].real / N_samples for m in 1..num_coeffs]
    b_m = -2 * [fft_result[m].imag / N_samples for m in 1..num_coeffs]
    return a0, a_m, b_m
```

**장점:**
- 계산 속도 $O(N\log N)$으로 매우 빠름 (vs. DFT의 $O(N^2)$)
- 모든 차수의 계수를 한 번에 계산
- 수치적으로 매우 안정적 (Gauss-Legendre의 끝점 특이성 문제 없음)
- **N_samples=2048**로 설정하여 높은 정확도

**단점:**
- 샘플 개수가 2의 거듭제곱이어야 함
- 메모리 사용량이 샘플 개수에 비례

**Gauss-Legendre vs. FFT:**
| 특징 | Gauss-Legendre | FFT |
|------|----------------|-----|
| 계산 속도 | 각 계수마다 별도 적분 필요 | 모든 계수 한 번에 ($O(N\log N)$) |
| 안정성 | 낮은 n에서 불안정 (끝점 특이성) | 모든 n에서 안정적 |
| 유연성 | 적분 구간 조정 가능 | 샘플링 고정 |
| 구현 난이도 | 간단 (节点/权重만 필요) | 복잡 (비트 역순, 버터플라이) |


## 可视化结果

