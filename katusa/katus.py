# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 17:46:12 2023

@author: 82103
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# 20개의 데이터와 해당하는 예측 값
data = np.array([[1830], [1729], [1840], [1780], [2040], [1920], [2100], [1930], [2100], [2070],
                 [2041], [2003], [2079], [2062], [1590], [1920]])
target = np.array([4.1, 7.5, 7.3, 5.2, 6.6, 8.9, 9.1, 9.4, 8.3, 8.1, 10.1, 9.2, 8.1, 7.6, 11.2, 7.9])

# 선형 회귀 모델 초기화
model = LinearRegression()

# 모델 훈련
model.fit(data, target)

# 예측을 위한 데이터 준비
new_data = np.array([[1790]])

# 예측 값 도출
predictions = model.predict(new_data)

# 결과 출력
for i in range(len(new_data)):
    print(f"Input: {new_data[i][0]}, Predicted Output: {predictions[i]}")
#%%
#MLR

temp_data = pd.read_csv('katus.csv')
 #print(temp_data.head())
data_origin = temp_data.to_numpy()

data_input = data_origin[:,2:].astype(np.float64)
data_target = data_origin[:, 1].astype(np.float64) #Q

anlz_x = data_input.copy()
anlz_y = data_target

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    anlz_x, anlz_y, random_state = 42)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 8, include_bias=False)

train_poly = poly.fit_transform(train_input)
print(train_poly.shape)

poly.get_feature_names_out()

test_poly = poly.transform(test_input)
print(test_poly.shape)

from sklearn.linear_model import LinearRegression

lr_mult = LinearRegression()

lr_mult.fit(train_poly, train_target)

lr_mult.fit(train_poly, train_target)

lr_mult.predict(train_poly)

score_test_mult = lr_mult.score(test_poly, test_target)
score_train_mult = lr_mult.score(train_poly, train_target)
print("Test set R2: {:.2f}".format(score_test_mult))
print("Training set R2: {:.2f}".format(score_train_mult))

predicted_value = lr_mult.predict(train_poly)

print("Predicted Value for the New Data:", predicted_value)
#%%
import numpy as np
from sklearn.linear_model import LinearRegression

# 해마다의 뽑는 수, 경쟁률 및 참여자 수 데이터 
years = np.array([1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16]) # 예시 해수
competition_rates = np.array([4.1, 7.5, 7.3, 5.2, 6.6, 8.9, 9.1, 9.4, 8.3, 8.1, 10.1, 9.2, 8.1, 7.6, 11.2, 7.9])
participants = np.array([[1830], [1729], [1840], [1780], [2040], [1920], [2100], [1930], [2100], [2070],
                 [2041], [2003], [2079], [2062], [1590], [1920]])

# 데이터를 reshape하여 모델에 맞게 변형
years = years.reshape(-1, 1)
competition_rates = competition_rates.reshape(-1, 1)
participants = participants.reshape(-1, 1)

# 선형 회귀 모델 초기화
model = LinearRegression()

# 모델 학습
model.fit(np.hstack((years, participants)), competition_rates)

# 다음 해(2023년)의 데이터로 예측을 수행
next_year = np.array([[17, 1762]])  # 다음 해의 해수와 예상 경쟁률
predicted_participants = model.predict(next_year)

print(f"2024년 예상 경쟁률: {predicted_participants[0]}")
#%%
data = data_origin[0:]
years = np.array(data[:,0])
competition_rates = np.array(data[:,1])
participants = np.array([[1420],[1830], [1729], [1840], [1780], [2040], [1920], [2100], [1930], [2100], [2070],
                 [2041], [2003], [2079], [2062],[1600], [1590], [1760], [1920]])

model = LinearRegression()
model.fit(np.hstack((years, participants)), competition_rates)

# 다음 해(2023년)의 데이터로 예측을 수행
next_year = np.array([[2024, 1762]])  # 다음 해의 해수와 예상 경쟁률
predicted_participants = model.predict(next_year)

print(f"2024년 예상 경쟁률: {predicted_participants[0]}")
#%%
import numpy as np
from sklearn.linear_model import LinearRegression


# 주어진 데이터
data = {
    #2005: [0, 0, 13.7, 6.1, 5.6, 5.2, 5.3, 5.2, 4.6, 4.1, 3.9, 3.9],
    #2006: [4.1, 5.9, 3.8, 3.5, 3.4, 3.4, 3.6, 3.6, 3.5, 3.5, 3.4, 3.5],
    #2007: [7.5, 7.8, 6.4, 5.6, 5.4, 5.1, 5.5, 5.5, 4.8, 4.4, 4.3, 5.4],
    #2008: [7.3, 7.8, 7.6, 6.6, 5.7, 5.1, 5.7, 5.5, 5.6, 5.1, 5.1, 4.7],
    #2009: [5.2, 5.3, 5.4, 5.3, 4.3, 4.2, 4.3, 4.3, 4.4, 4.2, 4, 4.1],
    #2010: [6.6, 6.8, 6.9, 6.5, 5.3, 4.6, 4.8, 4.8, 4.8, 4.6, 4.1, 4.1],
    2011: [8.9, 9, 9.1, 8.7, 7.3, 6.5, 6.7, 6.7, 6.9, 6.4, 6.1, 6.1],
    2012: [9.1, 9.2, 9.4, 9, 7.6, 6.8, 6.9, 7.1, 7.3, 6.9, 6.3, 6.4],
    2013: [9.4, 9.4, 9.5, 8.6, 7.4, 6.5, 6.8, 7.1, 7.5, 7, 6.3, 6.3],
    2014: [8.3, 8.4, 8.5, 8, 6.6, 5.6, 5.9, 6.1, 6.4, 5.9, 5.2, 5.2],
    2015: [8.1, 8.1, 8.4, 8.3, 8, 7.4, 7.4, 7.3, 7.3, 7, 6.6, 6.5],
    2016: [10.1, 10.7, 10.4, 9.8, 8.1, 6.9, 7, 7.3, 8, 7.3, 6.8, 6.8],
    2017: [9.2, 10.2, 13.6, 0, 10.2, 7.5, 7.4, 7.4, 7.4, 7, 6.1, 6],
    2018: [8.1, 8.4, 8.5, 8.1, 6.9, 6.3, 6.6, 6.8, 7.2, 7, 6.1, 6.1],
    2019: [7.6, 7.6, 7.7, 7.6, 7.2, 6.7, 6.8, 6.6, 6.5, 6.3, 6.2, 6.2],
    2020: [0, 19.2, 14.9, 13, 11.4, 10, 9.9, 9.4, 8.8, 8.2, 7.9, 7.9],
    2021: [11.2, 10.7, 9.6, 8.8, 8.1, 7.9, 8.8, 0, 8.2, 7.6, 7.4, 7.4],
    2022: [0, 11.2, 9.6, 8.8, 7.9, 7.1, 6.9, 6.7, 6.5, 6.4, 6.4, 6.4],
    2023: [7.9, 8.1, 0, 8.1, 7.6, 7.2, 7.2, 7.2, 7.1, 6.8, 6.8, 6.8]
}

# 참여자 수 데이터 (한 해의 데이터)
participants = np.array([
    #[1420], [1830], [1729], [1840], [1780], [2040], 
    [1920], [2100], [1930], [2100], [2070],
    [2041], [2003], [2079], [2062], [1600], [1590], [1760], [1920], [1792] # 2024년 데이터 추가
])

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# 데이터를 학습에 사용할 형태로 변환
years = []
months = []
values = []
participants_list = []

for year, monthly_values in data.items():
    for month, value in enumerate(monthly_values, start=1):
        years.append(year)
        months.append(month)
        values.append(value)
        participants_list.append(participants[year - 2011])

years = np.array(years).reshape(-1, 1)
months = np.array(months).reshape(-1, 1)
values = np.array(values)
participants_list = np.array(participants_list)

# 모델 학습
model = LinearRegression()
X = np.hstack((months, participants_list))
model.fit(X, values)

# 2024년 1월 값을 예측
prediction_jan_2024 = model.predict([[1, 1792]])

print(f"2024년 1월 예측값: {prediction_jan_2024[0]}")


for i in range(12):
    prediction_jan_2024 = model.predict([[i+1, 1792]])
    print(f"{i+1}월 : {prediction_jan_2024[0]}\n") 
    
    
    
   
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures



poly = PolynomialFeatures(degree =1 , include_bias=False)
X_poly = poly.fit_transform(np.hstack((months,participants_list)))

from sklearn.linear_model import LinearRegression

lr_mult = LinearRegression()
a = []
b = []
lr_mult.fit(X_poly, values)
important = np.array([[1,1792]])
important_poly = poly.transform(important)
for i in range(12):
    lr_mult.fit(X_poly, values)
    important = np.array([[i+1,1792]])
    important_poly = poly.transform(important)
    predicted_value = lr_mult.predict(important_poly)
    a.append(i+1)
    b.append(predicted_value)
    

print(predicted_value)

#%%

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    X, values, random_state = 42)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 2, include_bias=False)

train_poly = poly.fit_transform(train_input)
print(train_poly.shape)

poly.get_feature_names_out()

test_poly = poly.transform(test_input)
print(test_poly.shape)

from sklearn.linear_model import LinearRegression

lr_mult = LinearRegression()

lr_mult.fit(train_poly, train_target)

lr_mult.fit(train_poly, train_target)

lr_mult.predict(train_poly)

score_test_mult = lr_mult.score(test_poly, test_target)
score_train_mult = lr_mult.score(train_poly, train_target)
print("Test set R2: {:.2f}".format(score_test_mult))
print("Training set R2: {:.2f}".format(score_train_mult))

predicted_value = lr_mult.predict(train_poly)
import matplotlib.pyplot as plt
plt.plot(a, b, marker='o', linestyle='-')  # 선 그래프 그리기
plt.title('Results')  # 그래프 제목 설정
plt.xlabel('X-axis')  # X 축 레이블 설정
plt.ylabel('Y-axis')  # Y 축 레이블 설정
plt.grid(True)  # 그리드 표시
plt.show()  # 그래프 표시