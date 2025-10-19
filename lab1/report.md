<div style="display: flex; justify-content: space-between; font-size: 9px; margin-bottom: 10px;">
  <span>2021080070 计31 韩佑硕</span>
  <span><script>document.write(new Date().toISOString().split('T')[0])</script></span>
</div>

# 实验一 ： 傅里叶级数的可视化

# 代码分析

## `visualize()`
 전체 구조

  Fourier 급수를 회전하는 원들의 연쇄로 시각화합니다. 시간에 따라 원들이 회전하면서 마지막 원의 끝점이 원본 함수를 추적합니다.

  ---
  1. 준비 단계 (exp2.py:82-89)

  frames = 100  # 100개 프레임 (한 주기를 100단계로 나눔)

  # 원본 함수 그래프용 데이터
  x = np.linspace(0, 2π, 1000)  # 0~2π를 1000개 점
  y[i] = function(x[i])          # 각 점에서 f(t) 값

  for i in range(frames):
      time = 2π * i / 100  # 현재 시간 (0 → 2π)

  ---
  2. 데이터 구조 (exp2.py:88-89)

  point_pos_array = np.zeros((2*N_Fourier + 2, 2))  # 각 원의 끝점 위치 [x, y]
  radius_array = np.zeros(2*N_Fourier + 1)          # 각 원의 반지름

  인덱스 구조:  

  - [0]: 원점 (0, 0)
  - [1]: a0 원의 끝점
  - [2]: a1 원의 끝점
  - [3]: b1 원의 끝점
  - [4]: a2 원의 끝점
  - [5]: b2 원의 끝점
  - ...

  ---
  3. 첫 번째 원: a0 (exp2.py:91-96)

  point_pos_array[0] = [0, 0]        # 원의 중심 = 원점
  radius_array[0] = a0               # 반지름 = a0
  point_pos_array[1] = [0, a0]       # 원의 끝점 (시간 무관, 고정)

  - a0는 상수항 → 회전하지 않음
  - y축 방향으로 a0만큼 이동
  - 다음 원은 (0, a0)에서 시작

  ---
  4. a_n 원들 (exp2.py:99-105)

  for j in range(N_Fourier):  # j = 0, 1, 2, ...
      # a_{j+1} 원 (a1, a2, a3, ...)
      radius = fourier_coefficient(2*j + 1)  # n=1,3,5... → b1,b2,b3 아님 a1!

      # 이전 점에서 벡터 추가
      new_x = prev_x + radius * cos((j+1) * time)
      new_y = prev_y + radius * sin((j+1) * time)
      point_pos_array[2*j + 2] = [new_x, new_y]

  핵심:
  - 각속도 ω = (j+1) → 1, 2, 3, ...
  - 회전 벡터: radius × [cos(ωt), sin(ωt)]
  - cos 항: a_n × cos(nt)의 x, y 성분으로 분해
  - 이전 원의 끝에서 시작 → 벡터의 머리-꼬리 연결

  ---
  5. b_n 원들 (exp2.py:107-112)

      # b_{j+1} 원 (b1, b2, b3, ...)
      radius = fourier_coefficient(2*j + 2)  # n=2,4,6... → b1,b2,b3

      new_x = prev_x + radius * sin((j+1) * time)  # sin!
      new_y = prev_y + radius * cos((j+1) * time)  # cos!
      point_pos_array[2*j + 3] = [new_x, new_y]

  핵심:
  - sin/cos 순서 바뀜 → 90도 위상차
  - sin 항: b_n × sin(nt)의 x, y 성분
  - 반원파는 b_n = 0 → 이 원들은 반지름 0 (보이지 않음)

  ---
  6. 그래프 그리기 (exp2.py:114-117)

  # 1. 원들의 중심을 연결한 선
  plt.plot(point_pos_array[:, 0], point_pos_array[:, 1], 'o-')
  # → 원점 → (0,a0) → a1끝 → b1끝 → a2끝 → ... (파란선)

  # 2. 원본 함수
  plt.plot(x, y, '-')
  # → f(t) = √(π²-(t-π)²) 전체 그래프

  # 3. 빨간선: 근사 오차
  plt.plot([time, point_pos_array[-1][0]],
           [f_t, point_pos_array[-1][1]], '-', color='r')
  # → 실제 f(time)과 마지막 원 끝점 연결

  ---
  7. 핵심 원리

  마지막 원의 끝점 = (x_끝, y_끝)

  y_끝 = a0 + a1·cos(1·t) + b1·sin(1·t)
            + a2·cos(2·t) + b2·sin(2·t)
            + ...
       = Fourier 급수 근사값!

  - y_끝 ≈ f(t): N_Fourier가 클수록 정확
  - 빨간선 길이: 근사 오차
  - 시간 진행: 끝점이 원본 함수 궤적을 따라감

  ---
  8. 애니메이션 생성 (exp2.py:122-125)

  # 100개 프레임 → mp4 영상
  images = [imageio.imread(f"{i}.png") for i in range(100)]
  imageio.mimsave(f"{signal_name}_{N_Fourier}.mp4", images)

  결과: 원들이 회전하면서 반원파를 그려나가는 애니메이션!

# 任务一 ： 可视化方波信号

## 可视化结果

n이 비교적 작은 수일때는, 

# 任务二 ： 可视化半圆波信号
## 可视化结果

