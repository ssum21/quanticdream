# 📈 회귀분석 마스터하기 01

## 1. 개요: 회귀분석의 세계로

이번에는 회귀분석(Regression Analysis)이라는 중요한 통계적 방법에 대해 배웁니다. 회귀분석은 데이터 사이언스, 기계 학습, 경제학 등 다양한 분야에서 변수들 사이의 관계를 모델링하고 예측하는 데 사용하는 핵심적인 도구입니다.

이 교재를 통해 회귀분석의 기본 개념부터 실제 데이터에 코드를 적용하는 방법까지, 모든 것을 배우게 될 것입니다.

### 학습 목표

> *   회귀분석의 기본 개념과 작동 원리를 이해합니다.
> *   데이터가 회귀분석에 적합한지 기본 가정을 검증할 수 있습니다.
> *   Python 코드를 사용하여 회귀 모델을 구축하고 평가할 수 있습니다.
> *   모델의 결과를 해석하고, 통계적 유의성을 판단할 수 있습니다.

---

## 2. 스테이지 1: 회귀분석 핵심 개념

### 2.1. 회귀분석이란? 🤔

회귀분석은 변수들 사이의 **관계를 이해하고 예측**하는 데 도움을 주는 통계적 방법입니다. 간단히 말해, 하나의 변수(**종속 변수**, 우리가 예측하고 싶은 것)가 다른 하나 또는 그 이상의 변수들(**독립 변수**)에 의해 어떻게 영향을 받는지를 수치적으로 모델링하는 기법입니다.

> **예시: 광고비와 매출의 관계**
> 기업이 광고에 지출하는 비용(독립 변수)과 제품 매출(종속 변수) 사이의 관계가 궁금하다고 가정해 봅시다. 회귀분석을 사용하면, "광고비를 100만 원 늘리면 매출이 평균적으로 얼마나 증가하는가?"와 같은 질문에 답할 수 있습니다.

회귀분석에는 여러 종류가 있지만, 가장 기본적인 두 가지는 다음과 같습니다.

*   **단순 선형 회귀 (Simple Linear Regression)**
    *   하나의 독립 변수가 하나의 종속 변수에 미치는 영향을 분석합니다. (예: `매출` = a + b * `광고비`)
*   **다중 선형 회귀 (Multiple Linear Regression)**
    *   여러 개의 독립 변수가 하나의 종속 변수에 미치는 영향을 분석합니다. (예: `매출` = a + b1 * `광고비` + b2 * `시장 규모` + b3 * `계절 요인`)

### 2.2. 언제 사용하면 좋을까요?

*   **작은 데이터 세트에서도 효과적**: 복잡한 머신러닝 모델과 달리, 상대적으로 작은 데이터로도 의미 있는 결과를 도출할 수 있습니다.
*   **이해와 설명이 중요할 때**: 예측 정확도만큼 "왜" 그런 결과가 나왔는지 설명하는 것이 중요할 때 강력합니다. 각 변수가 결과에 미치는 영향을 명확히 보여줍니다.
*   **선형 관계가 강한 데이터**: 변수 간에 직선과 같은 관계가 보일 때 높은 정확도를 제공합니다.
*   **간단하고 비용이 적은 모델**: 모델이 단순하여 계산 속도가 빠르고, 자원 소모가 적어 신속한 분석에 유리합니다.

### 2.3. 한계점

*   **선형 관계에만 집중 📏**: 실제 세계의 복잡한 곡선(비선형) 관계를 제대로 표현하지 못할 수 있습니다.
*   **단순함의 한계 🤔**: 모델이 단순하기 때문에, 데이터의 매우 복잡하고 미묘한 패턴을 놓칠 수 있습니다.

### 2.4. 회귀분석의 이론적 배경

단순 선형 회귀분석의 식은 다음과 같습니다.

> ### $Y = a + bX + \epsilon$

*   `Y`: **종속 변수 (Dependent Variable)** - 우리가 예측하려는 목표 변수입니다. (예: 집값)
*   `X`: **독립 변수 (Independent Variable)** - Y에 영향을 준다고 생각하는 변수입니다. (예: 집의 평수)
*   `b`: **회귀 계수 (Coefficient)** - 기울기(slope)입니다. X가 1단위 증가할 때 Y가 평균적으로 얼마나 변하는지를 나타냅니다.
*   `a`: **절편 (Intercept)** - X가 0일 때의 Y값입니다.
*   `ϵ`: **오차항 (Error Term)** - 우리 모델이 설명하지 못하는 무작위적인 변동이나 노이즈를 의미합니다.

회귀분석의 목표는 실제 데이터 포인트들과의 거리(오차)의 제곱합이 최소가 되는 가장 이상적인 직선(회귀선)을 찾는 것이며, 이 방법을 **최소제곱법(Ordinary Least Squares, OLS)**이라고 합니다.

---

## 3. 스테이지 2: Statsmodels로 회귀모델 만들기 (실습)

이제 Python의 `Statsmodels` 라이브러리를 사용하여 직접 회귀 모델을 만들어 보겠습니다.

### 3.1. 1단계: 데이터 준비

먼저, 연습용 데이터를 생성합니다. `numpy`를 사용하면 간단히 만들 수 있습니다.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 재현성을 위해 시드(seed) 설정
np.random.seed(42)

# 데이터 생성
# X: 독립 변수 (0~100 사이의 값 100개)
X = np.random.rand(100, 1) * 100
# y: 종속 변수 (y = 50 + 2*X + noise 형태의 관계)
y = 50 + 2 * X + np.random.randn(100, 1) * 30

# 생성된 데이터 시각화
plt.scatter(X, y, color='skyblue', label='Data Points')
plt.title('Generated Data for Regression')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (y)')
plt.legend()
plt.grid(True)
plt.show()
```

### 3.2. 2단계: 모델 학습

`statsmodels`를 사용하여 OLS 회귀 모델을 학습시킵니다.

#### `sm.add_constant()`: 상수항(절편) 추가하기
회귀식 $Y = a + bX$ 에서 절편 `a`를 모델이 학습할 수 있도록, 데이터에 상수항(보통 1로 채워진 열)을 추가해주어야 합니다. `sm.add_constant()` 함수가 이 역할을 합니다.

```python
# 1. 상수항 추가
X_const = sm.add_constant(X)

# 데이터 확인 (첫 5개)
# print(X_const[:5])
# [[  1.     37.45401188]
#  [  1.     95.07143064]
#  [  1.     73.19939418]
#  [  1.     59.86584842]
#  [  1.     15.60186404]]

# 2. OLS 모델 생성 및 학습
# sm.OLS(종속변수, 독립변수)
model = sm.OLS(y, X_const)
results = model.fit()
```

### 3.3. 3단계: 모델 결과 분석 및 예측

모델 학습이 완료되었습니다! `results.summary()` 함수는 모델의 모든 통계 정보를 요약하여 보여주는 강력한 도구입니다.

```python
# 모델 요약 결과 출력
print(results.summary())
```

#### 결과 시각화
학습된 회귀선이 데이터를 얼마나 잘 설명하는지 시각적으로 확인해 봅시다.

```python
# 예측값 생성
y_pred = results.predict(X_const)

# 원본 데이터 산점도
plt.scatter(X, y, color='skyblue', label='Actual Data')
# 회귀선 그리기
plt.plot(X, y_pred, color='lightcoral', linewidth=2, label='Regression Line')

plt.title('Simple Linear Regression with Statsmodels')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (y)')
plt.legend()
plt.grid(True)
plt.show()
'''

---

## 4. 스테이지 3: 모델의 건강 진단하기 - 결과 해석과 가정 검증 🩺

모델을 만드는 것은 절반의 성공일 뿐입니다. 이제 우리는 모델이 얼마나 좋은지, 그리고 그 결과를 믿을 수 있는지 확인하는 '건강 검진'을 시작해야 합니다. 이 단계에서는 `statsmodels`가 제공하는 강력한 요약 리포트를 해석하고, 회귀분석의 핵심적인 기본 가정들이 지켜졌는지 진단하는 방법을 배웁니다.

### 4.1. 모델 성적표 열어보기: `summary()` 결과 완벽 해부

`results.summary()` 함수는 우리 모델의 모든 것을 알려주는 상세한 성적표와 같습니다. 이 성적표는 크게 두 부분으로 나눌 수 있습니다: **(1) 모델 전체의 성능**과 **(2) 각 변수(계수)의 개별 성능**입니다.

```python
# 이전 단계에서 학습된 모델의 요약 결과 출력
print(results.summary())
```

<details>
<summary>📋 `results.summary()` 출력 예시 (클릭하여 확인)</summary>

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.812
Model:                            OLS   Adj. R-squared:                  0.810
Method:                 Least Squares   F-statistic:                     423.4
Date:                Sat, 20 Jul 2024   Prob (F-statistic):           1.33e-35
Time:                        12:00:00   Log-Likelihood:                -482.63
No. Observations:                 100   AIC:                             969.3
Df Residuals:                      98   BIC:                             974.5
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         49.3486      4.672     10.563      0.000      40.077      58.620
x1             2.0306      0.099     20.577      0.000       1.835       2.226
==============================================================================
Omnibus:                        0.320   Durbin-Watson:                   1.936
Prob(Omnibus):                  0.852   Jarque-Bera (JB):                0.485
Skew:                          -0.034   Prob(JB):                        0.785
Kurtosis:                       2.716   Cond. No.                         111.
==============================================================================
```
</details>

#### 파트 1: 모델의 전체적인 합격 여부 (모델 적합도 📊)

요약표의 **오른쪽 상단**은 모델이 전반적으로 데이터를 얼마나 잘 설명하는지를 보여줍니다.

| 지표 | 의미 | 해석 가이드 |
| :--- | :--- | :--- |
| **R-squared** | **설명력**: 모델이 데이터의 변동성 중 몇 퍼센트(%)를 설명하는지를 나타냅니다. (0~1 사이 값) | **0.812**는 우리 모델이 종속 변수(y)의 움직임 중 약 **81.2%**를 설명한다는 의미입니다. 일반적으로 높을수록 좋습니다. |
| **Adj. R-squared** | **보정된 설명력**: 불필요한 변수가 추가될 때 R-squared가 무조건 오르는 것을 방지하기 위해 보정한 값입니다. | 다중 회귀분석에서 여러 모델을 비교할 때 이 지표를 보는 것이 더 정확합니다. |
| **F-statistic** | **모델의 유의성**: "과연 이 모델이 아무 의미도 없는 쓰레기 모델은 아닐까?"를 검증합니다. | 이 값이 충분히 커야 합니다. (기준은 데이터마다 다름) |
| **Prob (F-statistic)** | **모델의 신뢰도 (p-value)**: F-통계량에 대한 p-value입니다. | **0.05보다 작으면** "이 모델은 통계적으로 유의미하다"고 자신 있게 말할 수 있습니다. 1.33e-35는 0에 매우 가까우므로 합격입니다. |
| **AIC / BIC** | **모델의 좋음**: 모델의 복잡도와 설명력을 동시에 고려한 지표입니다. | 절대적인 기준은 없으며, 여러 모델을 비교할 때 **더 낮은 값을 가진 모델**을 선택합니다. |

> **중간 결론**: 우리 모델은 설명력(R-squared)이 높고, 통계적으로 매우 유의미(Prob (F-statistic) < 0.05)하므로, **전체적으로 합격**입니다.

#### 파트 2: 각 변수의 영향력과 신뢰도 (계수 분석 📈)

요약표의 **가운데 부분**은 각 독립 변수가 종속 변수에 얼마나, 그리고 얼마나 확실하게 영향을 미치는지를 보여줍니다.

| 지표 | 의미 | 해석 가이드 (`x1` 변수 기준) |
| :--- | :--- | :--- |
| **`coef`** | **영향력의 크기 (회귀 계수)**: 해당 변수가 1단위 변할 때 종속 변수가 평균적으로 얼마나 변하는지 나타냅니다. | `x1`의 `coef`가 **2.0306**이므로, "X가 1 증가할 때 y는 평균적으로 약 2.03 증가한다"고 해석할 수 있습니다. `const`는 절편(a)입니다. |
| **`std err`** | **추정의 불확실성 (표준 오차)**: `coef` 추정치가 얼마나 변동할 수 있는지를 나타냅니다. | 작을수록 `coef` 값을 더 신뢰할 수 있습니다. |
| **`t`** | **영향력의 상대적 크기 (t-통계량)**: `coef` 값을 `std err`로 나눈 값입니다. | 이 값의 절댓값이 클수록 해당 변수가 더 중요하다고 봅니다. (일반적으로 2 이상이면 유의미) |
| **`P>t`** | **영향력의 신뢰도 (p-value)**: "이 변수의 영향력(`coef`)이 실제로는 0인데, 우연히 0이 아닌 것처럼 나온 것은 아닐까?"에 대한 답입니다. | **0.05보다 작으면**, "이 변수의 영향력은 우연이 아니며, 통계적으로 유의미하다"고 결론 내릴 수 있습니다. `x1`과 `const` 모두 0.000이므로 매우 유의미합니다. |

> **중간 결론**: 독립 변수 `x1`은 종속 변수 `y`에 **통계적으로 매우 유의미한 양(+)의 영향**을 미칩니다.

### 4.2. 보이지 않는 위험 찾기: 회귀분석의 4대 기본 가정 검증 🔍

좋은 성적표(summary)를 받았더라도, 그 결과가 신뢰할 수 있으려면 데이터가 회귀분석의 기본 가정을 만족해야 합니다. 이 가정이 깨지면 모델의 해석이 왜곡될 수 있습니다.

#### 가정 1: 선형성 (Linearity)

*   **의미**: 독립 변수와 종속 변수 간의 관계가 직선적이어야 합니다.
*   **중요성**: 관계가 곡선 형태인데 직선으로 모델링하면 예측이 부정확해집니다.
*   **진단 방법**: **예측값 vs 실제값 산점도**를 그려봅니다. 점들이 45도 대각선 주위에 무작위로 분포하면 이상적입니다.

```python
# 실제값(y)과 모델의 예측값(results.fittedvalues)을 비교
plt.scatter(y, results.fittedvalues, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Fit')
plt.title('Actual vs. Predicted Values Plot')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.legend()
plt.show()
```

> **진단 결과**: 점들이 붉은 점선 주변에 비교적 잘 모여 있으므로, **선형성 가정을 만족**한다고 볼 수 있습니다.

#### 가정 2: 오차의 정규성 (Normality of Errors)

*   **의미**: 모델이 예측하지 못하는 오차(잔차, Residuals)들이 정규분포를 따라야 합니다.
*   **중요성**: 이 가정이 만족되어야 계수의 t-검정, 모델의 F-검정 등 통계적 검증 결과를 신뢰할 수 있습니다.
*   **진단 방법**:
    1.  **Q-Q Plot**: 점들이 붉은 대각선 위에 거의 일직선으로 놓여있으면 정규성을 만족합니다.
    2.  **Shapiro-Wilk Test**: 통계 검정. p-value가 0.05보다 크면 "정규성을 따른다"는 귀무가설을 기각할 수 없으므로 정규성을 만족한다고 봅니다.

```python
from scipy import stats

# 잔차 계산
residuals = results.resid

# 1. Q-Q Plot
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()

# 2. Shapiro-Wilk Test
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"Shapiro-Wilk Test: Statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
```

> **진단 결과**: Q-Q plot에서 점들이 대각선에 잘 붙어 있고, Shapiro 검정의 p-value가 0.05보다 크므로 **오차의 정규성 가정을 만족**합니다.

#### 가정 3: 오차의 등분산성 (Homoscedasticity)

*   **의미**: 예측값의 크기와 상관없이 오차의 분산(흩어진 정도)이 일정해야 합니다.
*   **중요성**: 분산이 일정하지 않으면(이분산성), 특정 구간에서는 예측을 잘하고 다른 구간에서는 예측을 못하는 불안정한 모델이 됩니다.
*   **진단 방법**: **예측값 vs 잔차 산점도**를 그려봅니다. 점들이 y=0 선을 기준으로 특별한 패턴(예: 깔때기 모양) 없이 무작위로 흩뿌려져 있어야 합니다.

```python
# 예측값(fittedvalues) vs 잔차(residuals) 산점도
plt.scatter(results.fittedvalues, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.title('Residuals vs. Fitted Values Plot')
plt.xlabel('Fitted Values (Predicted)')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()
```

> **진단 결과**: 점들이 0선을 기준으로 특별한 패턴 없이 고르게 분포하고 있습니다. 따라서 **오차의 등분산성 가정을 만족**합니다.

#### 가정 4: 다중공선성 (Multicollinearity) 부재 - (다중 회귀분석 시)

*   **의미**: 독립 변수들끼리 너무 강한 상관관계를 가지면 안 됩니다.
*   **중요성**: 이 문제가 발생하면, 어떤 변수가 진짜 원인인지 파악하기 어려워지고 계수 추정치가 불안정해집니다.
*   **진단 방법**: **분산팽창계수(VIF, Variance Inflation Factor)**를 계산합니다. 일반적으로 VIF가 10을 넘으면 다중공선성 문제가 있다고 판단합니다. (현재는 독립 변수가 1개이므로 해당 사항 없음)

```python
# 이 코드는 독립변수가 2개 이상인 다중 회귀분석에서 사용합니다.
# from statsmodels.stats.outliers_influence import variance_inflation_factor

# # X_multi가 독립 변수들로만 구성된 DataFrame이라고 가정
# vif = pd.DataFrame()
# vif["VIF Factor"] = [variance_inflation_factor(X_multi.values, i) for i in range(X_multi.shape[1])]
# vif["features"] = X_multi.columns
# print(vif)
```