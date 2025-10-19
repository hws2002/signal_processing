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

### 1. Gauss-Legendre
먼저 쓴 방법은 Gauss-Legendre방식이다. 이는 节点과 权重만 있으면 구현이 매우 쉬운 편이지만, 节点이 150개  정도 있어야 `N_Fourier=128` 인 경우에도 안정적인 근사값을 구해낼 수 있었다.

### 2. FFT


## 可视化结果

