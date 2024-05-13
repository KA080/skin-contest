import pandas as pd
df = pd.read_csv('학습용데이터.csv')
# X와 y는 각각 피처와 타겟 변수를 나타냅니다.
x = df[['age','m_e_w',	'm_f_w',	'm_c_a',	'm_f_a',	'm_m',	'm_nm',	'm_c_o',	'm_f_o']]
y = df[['an_e_w_rou',	'an_e_w_dep',	'an_f_w_rou',	'an_f_w_dep',	'seb_c_o',	'seb_f_o',	'cor_c_a',	'cor_f_a',	'mex_m',	'mex_nm']]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures


# 학습 데이터와 테스트 데이터로 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 선형회귀 모델 생성
model = LinearRegression()

# 표준화 (StandardScaler 사용)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 모델 학습 및 예측
model.fit(x_train_scaled, y_train)
y_pred_scaled = model.predict(x_test_scaled)

# 모델 평가 (예: 평균 제곱 오차)
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
print(f'Scaled Mean Squared Error: {mse_scaled}')

# 교차 검증
cv_scores = cross_val_score(model, x_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()
print(f'Cross-validated Mean Squared Error: {cv_mse}')

# 다항 특성 추가
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train_scaled)
x_test_poly = poly.transform(x_test_scaled)

# 모델 학습 및 예측
model.fit(x_train_poly, y_train)
y_pred_poly = model.predict(x_test_poly)

# 모델 평가
mse_poly = mean_squared_error(y_test, y_pred_poly)
print(f'Polynomial Mean Squared Error: {mse_poly}')

