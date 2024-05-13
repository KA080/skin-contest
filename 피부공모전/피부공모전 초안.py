import pandas as pd
df = pd.read_csv('학습용데이터.csv')
# X와 y는 각각 피처와 타겟 변수를 나타냅니다.
x = df[['age','m_e_w',	'm_f_w',	'm_c_a',	'm_f_a',	'm_m',	'm_nm',	'm_c_o',	'm_f_o']]
y = df[['an_e_w_rou',	'an_e_w_dep',	'an_f_w_rou',	'an_f_w_dep',	'seb_c_o',	'seb_f_o',	'cor_c_a',	'cor_f_a',	'mex_m',	'mex_nm']]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


# 학습 데이터와 테스트 데이터로 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 선형회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x_train, y_train)

# 학습된 모델을 사용하여 테스트 데이터에 대한 예측 수행
y_pred = model.predict(x_test)

# 모델 평가 (예: 평균 제곱 오차)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 전체 데이터셋에 대한 상관 행렬 계산
correlation_matrix_all = df.corr()

# heatmap을 통해 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_all, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix (Full Dataset)')
plt.show()