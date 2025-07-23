# 📈 머신러닝 회귀 프로젝트 실전 가이드 (Pandas & Scikit-learn)

이 문서는 데이터 분석 및 머신러닝 프로젝트의 표준적인 흐름을 처음부터 끝까지 안내하는 실전 가이드입니다. 데이터 불러오기부터 탐색, 전처리, 모델링, 평가, 그리고 최종 결과 제출까지의 전 과정을 상세히 다룹니다.

## 1단계: 데이터 준비 및 탐색 (EDA - Exploratory Data Analysis)

모든 프로젝트의 시작은 데이터를 이해하는 것에서 출발합니다. 어떤 데이터가 있는지, 결측치는 없는지, 각 변수는 어떤 의미를 갖는지 파악하는 단계입니다.

### 1.1. 라이브러리 불러오기

먼저, 데이터 분석에 필요한 필수 도구들(라이브러리)을 불러옵니다.

*   `pandas`: 표 형태의 데이터를 다루는 데 최적화된 도구입니다. (별명: `pd`)
*   `numpy`: 수치 계산, 특히 배열(행렬) 연산을 위한 도구입니다. (별명: `np`)
*   `matplotlib.pyplot` / `seaborn`: 데이터를 시각화하여 인사이트를 얻는 도구입니다. (별명: `plt`, `sns`)

```python
# 데이터 분석 및 시각화를 위한 필수 라이브러리를 불러옵니다.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### 1.2. 데이터 불러오기

`pandas`의 `read_csv()` 함수를 사용하여 `train.csv`, `test.csv`와 같은 데이터 파일을 **데이터프레임(DataFrame)**이라는 표 형태로 불러옵니다.

```python
# 학습(train), 테스트(test), 제출(submission)용 CSV 파일을 불러옵니다.
raw_data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')
```

### 1.3. 원본은 소중하게! - 깊은 복사(Deep Copy)

불러온 원본 데이터(`raw_data`)를 직접 수정하면, 실수했을 때 처음부터 다시 불러와야 하는 번거로움이 있습니다. 이를 방지하기 위해 `.copy()` 메서드를 사용하여 **깊은 복사(Deep Copy)**를 수행합니다.

*   **얕은 복사 (Shallow Copy)**: 원본 데이터의 주소만 복사합니다. 복사본을 수정하면 원본도 함께 변경됩니다.
*   **깊은 복사 (Deep Copy)**: 데이터 자체를 완전히 새로운 객체로 복사합니다. 복사본을 수정해도 원본은 안전하게 유지됩니다.



```python
# 원본 데이터를 보호하기 위해 깊은 복사를 통해 작업용 데이터프레임을 생성합니다.
train = raw_data.copy()
```

### 1.4. 데이터 훑어보기

#### `head()`: 데이터의 첫인상 확인

데이터의 최상위 5개 행을 출력하여 전체적인 구조(컬럼, 값)를 파악합니다.

```python
# 데이터의 상위 3개 행을 확인합니다.
train.head(3)
```

#### `info()`: 데이터의 요약 정보(이력서) 확인

각 컬럼의 정보(데이터 타입, 결측치가 아닌 값의 개수 등)를 한눈에 파악할 수 있어 매우 유용합니다.

```python
# 데이터프레임의 전체적인 정보를 요약하여 출력합니다.
train.info()
```

<details>
<summary>📋 `info()` 결과 해석하기</summary>

*   **`RangeIndex`**: 총 데이터 행(row)의 개수를 알려줍니다.
*   **`Data columns`**: 총 컬럼(column)의 개수를 알려줍니다.
*   **`Non-Null Count`**: 각 컬럼별로 비어있지 않은(non-null) 데이터의 개수입니다. 이 값이 전체 행의 개수보다 작다면, **결측치(Missing Value)**가 있다는 의미입니다.
*   **`Dtype`**: 각 컬럼의 데이터 타입입니다. (예: `int64`(정수), `float64`(실수), `object`(문자열))
*   **`memory usage`**: 데이터프레임이 메모리에서 차지하는 용량을 보여줍니다.
</details>

#### `describe()`: 데이터의 기술 통계량 확인

수치형 데이터 컬럼들의 핵심 통계량을 요약하여 보여줍니다. 데이터의 분포를 파악하는 데 필수적입니다.

```python
# 수치형 컬럼들의 기술 통계량을 확인합니다.
train.describe()
```

<details>
<summary>📊 `describe()` 결과 해석하기</summary>

*   **`count`**: 데이터 개수
*   **`mean`**: 평균값
*   **`std`**: 표준편차. 데이터가 평균으로부터 얼마나 퍼져 있는지를 나타냅니다.
*   **`min`**: 최솟값
*   **`25%` / `50%` / `75%`**: 각각 1사분위수, 2사분위수(중앙값), 3사분위수를 의미합니다. 데이터의 분포를 파악하는 데 사용됩니다.
*   **`max`**: 최댓값
</details>

## 2단계: 데이터 시각화를 통한 인사이트 발견

숫자로만 된 통계량은 직관적으로 이해하기 어렵습니다. 데이터를 그림으로 표현하면 숨겨진 패턴과 관계를 쉽게 발견할 수 있습니다.

### 2.1. 타겟 변수 분포 확인: `histplot()`

우리가 예측해야 할 목표, 즉 **타겟 변수(`TARGET`)**의 분포를 히스토그램으로 확인합니다. 데이터가 특정 값에 치우쳐 있는지, 정규분포를 따르는지 등을 파악할 수 있습니다.

```python
# 타겟 변수의 분포를 히스토그램으로 시각화합니다.
sns.histplot(data=train, x='TARGET')
plt.title('Target Variable Distribution')
plt.show()
```

### 2.2. 변수 간 상관관계 분석

#### `regplot()`: 개별 변수와 타겟의 관계 파악

산점도와 회귀선을 함께 그려 두 변수 간의 관계(양의 관계, 음의 관계, 무관계)를 시각적으로 파악합니다.

```python
# WS(풍속) 변수와 TARGET 간의 관계를 산점도와 회귀선으로 확인합니다.
sns.regplot(data=train, x='WS', y='TARGET')
plt.title('Relationship between Wind Speed and Target')
plt.show()
```

#### `heatmap()`: 모든 변수 간 상관관계 한눈에 보기

상관관계 행렬을 색상으로 표현한 히트맵을 사용하면, 어떤 변수들이 서로 강한 관계를 맺고 있는지 전체적으로 파악할 수 있습니다.

*   **`annot=True`**: 각 셀에 상관계수 값을 표시합니다.
*   값이 1에 가까울수록 강한 양의 상관관계, -1에 가까울수록 강한 음의 상관관계를 의미합니다.

```python
# 모든 수치형 변수들 간의 상관관계를 계산합니다.
correlation_matrix = train.corr()

# 히트맵으로 시각화합니다.
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()
```

## 3단계: 모델링을 위한 데이터 전처리

모델이 데이터를 잘 학습할 수 있도록 데이터를 "정제"하는 과정입니다.

### 3.1. 독립 변수와 종속 변수 분리

*   **독립 변수 (X)**: 예측에 사용할 재료. (Feature)
*   **종속 변수 (y)**: 우리가 예측하고자 하는 목표. (Target)

`drop()` 메서드를 사용하여 원본 데이터에서 불필요한 컬럼(`ID`)과 타겟 컬럼(`TARGET`)을 제거하여 `train_x`를 만듭니다.

```python
# 'TARGET'과 불필요한 'ID' 컬럼을 제외하여 독립 변수(train_x)를 만듭니다.
train_x = train.drop(columns=['ID', 'TARGET'])

# 'TARGET' 컬럼만 선택하여 종속 변수(train_y)를 만듭니다.
train_y = train['TARGET']
```

### 3.2. 학습/검증 데이터 분리: `train_test_split`

모델의 성능을 객관적으로 평가하기 위해, 보유한 데이터를 **학습용(train) 데이터**와 **검증용(validation) 데이터**로 나눕니다.

*   **학습용 데이터**: 모델을 학습시키는 데 사용 (교과서)
*   **검증용 데이터**: 학습된 모델의 성능을 평가하는 데 사용 (모의고사)
*   `test_size`: 전체 데이터 중 검증용으로 사용할 비율을 지정합니다. (예: 0.2 = 20%)
*   `random_state`: 데이터를 나눌 때 무작위 샘플링을 하는데, 이 값을 고정하면 항상 동일한 방식으로 데이터가 나뉘어 실험 결과를 재현할 수 있습니다.

```python
from sklearn.model_selection import train_test_split

# 데이터를 학습용 80%, 검증용 20%로 분리합니다.
X_train, X_valid, y_train, y_valid = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42
)
```

## 4단계: 머신러닝 모델 구축 및 학습

이제 준비된 데이터를 사용하여 다양한 머신러닝 모델을 만들고 학습시킵니다.

### 4.1. 모델 정의 및 학습

다양한 회귀 모델(DecisionTree, RandomForest, XGBoost 등)을 `scikit-learn` 라이브러리를 통해 쉽게 정의하고, `.fit()` 메서드를 사용하여 학습시킵니다.

#### 의사결정 나무 (Decision Tree)
데이터를 특정 기준으로 나누는 질문들을 반복하여 예측하는 모델. 단순하고 해석이 쉽지만 과적합(overfitting) 위험이 있습니다.

#### 랜덤 포레스트 (Random Forest)
여러 개의 의사결정 나무를 만들어 그 예측 결과를 종합(평균)하는 **앙상블(Ensemble)** 모델입니다. 단일 나무보다 훨씬 안정적이고 성능이 뛰어납니다.



```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# 1. 의사결정 나무 모델 정의 및 학습
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)

# 2. 랜덤 포레스트 모델 정의 및 학습
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)

# 3. XGBoost 모델 정의 및 학습
model_xgb = XGBRegressor(random_state=42)
model_xgb.fit(X_train, y_train)
```

## 5단계: 모델 성능 평가 및 피처 중요도 확인

### 5.1. 모델 성능 비교

검증용 데이터(`X_valid`, `y_valid`)를 사용하여 각 모델의 예측 성능을 평가합니다. 회귀 문제에서는 주로 **MSE(Mean Squared Error)**나 **RMSE(Root Mean Squared Error)**를 사용합니다.

```python
from sklearn.metrics import mean_squared_error

# 각 모델을 사용하여 검증 데이터에 대한 예측을 수행합니다.
pred_dt = model_dt.predict(X_valid)
pred_rf = model_rf.predict(X_valid)
pred_xgb = model_xgb.predict(X_valid)

# 각 모델의 MSE를 계산합니다.
mse_dt = mean_squared_error(y_valid, pred_dt)
mse_rf = mean_squared_error(y_valid, pred_rf)
mse_xgb = mean_squared_error(y_valid, pred_xgb)

print(f"Decision Tree MSE: {mse_dt:.4f}")
print(f"Random Forest MSE: {mse_rf:.4f}")
print(f"XGBoost MSE: {mse_xgb:.4f}")
```

### 5.2. 피처(Feature) 중요도 확인

랜덤 포레스트와 같은 트리 기반 모델은 어떤 변수(피처)가 예측에 더 중요한 역할을 했는지 알려주는 **`feature_importances_`** 속성을 제공합니다. 이는 모델을 해석하고, 피처 선택(Feature Selection)에 활용할 수 있는 중요한 정보입니다.

```python
# 랜덤 포레스트 모델의 피처 중요도를 가져옵니다.
importances = model_rf.feature_importances_
feature_names = train_x.columns

# 피처 중요도를 시각화합니다.
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importances from Random Forest')
plt.show()```

## 6단계: 최종 예측 및 제출 파일 생성

가장 성능이 좋았던 모델을 선택하여, 실제 문제인 `test` 데이터에 대한 예측을 수행하고 제출 형식에 맞게 파일을 저장합니다.

```python
# 테스트 데이터에서 불필요한 'ID' 컬럼을 제거합니다.
test_x = test.drop(columns=['ID'])

# 가장 성능이 좋았던 모델(예: 랜덤 포레스트)로 최종 예측을 수행합니다.
final_pred = model_rf.predict(test_x)

# 제출 파일(submission.csv)의 'TARGET' 컬럼에 예측값을 채워넣습니다.
submission['TARGET'] = final_pred

# 최종 제출 파일을 저장합니다. index=False 옵션은 불필요한 인덱스 열을 제거합니다.
submission.to_csv('my_submission.csv', index=False)

# 저장된 파일 확인
submission.head()
```