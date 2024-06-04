# 필요한 라이브러리 임포트
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import plotly.express as px

import category_encoders as ce
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# 모든 무작위성을 통제하기 위한 시드 설정 함수 정의
def seed_everything(seed):
    # random 모듈의 시드 설정
    random.seed(seed)
    # PYTHONHASHSEED 환경 변수 설정
    os.environ['PYTHONHASHSEED'] = str(seed)
    # numpy 모듈의 시드 설정
    np.random.seed(seed)

# 시드 값 설정
seed_everything(42)

# 'train.csv' 파일에서 처음 1000개의 행만 로드
train = pd.read_csv('train.csv', nrows=1000)
# 'test.csv' 파일 로드
test = pd.read_csv('test.csv')

# 데이터의 첫 5행을 출력하여 데이터 확인
train.head()

# 'Click' 열의 값 분포를 확인 (비율 계산)
click = train['Click'].value_counts(normalize=True)

# 'Click' 값 분포를 바 그래프로 시각화
click_figure = px.bar(
    click,  # 데이터
    x=['Not Clicked : 0', 'Clicked : 1'],  # x축 값
    y=click.values.tolist(),  # y축 값
    labels={'x': 'Value', 'y': 'Percentage'},  # 축 라벨
    width=450,  # 그래프 너비
    height=500  # 그래프 높이
)
click_figure.show()

# 훈련 데이터에서 ID와 Click 열 제거하여 피처와 타겟 변수로 분리
train_x = train.drop(columns=['ID', 'Click'])
train_y = train['Click']

# 테스트 데이터에서 ID 열 제거
test_x = test.drop(columns=['ID'])

# 각 열에 대해 결측값을 확인하고, 결측값이 있는 경우 0으로 대체
for col in tqdm(train_x.columns):
    if train_x[col].isnull().sum() != 0:
        train_x[col].fillna(0, inplace=True)
        test_x[col].fillna(0, inplace=True)

# 범주형 변수 인코딩 (카운트 인코딩 사용)
# 데이터 타입이 object인 열 목록을 추출
encoding_target = list(train_x.dtypes[train_x.dtypes == "object"].index)

# 카운트 인코더를 사용하여 범주형 변수 인코딩
enc = ce.CountEncoder(cols=encoding_target).fit(train_x, train_y)
X_train_encoded = enc.transform(train_x)
X_test_encoded = enc.transform(test_x)

# 다양한 분류기 초기화
ada_boosting = AdaBoostClassifier()
random_forest = RandomForestClassifier()
gradient_boosting = GradientBoostingClassifier()

# 랜덤 포레스트 모델 설정 및 하이퍼파라미터 그리드 서치 설정
rf_model = RandomForestClassifier()
rf_param_grid = {
    'n_estimators': [900],  # 트리의 개수
    'max_features': ['sqrt'],  # 각 트리의 최대 피처 개수
    'max_depth': [95]  # 트리의 최대 깊이
}
rf_grid_search = GridSearchCV(
    estimator=rf_model,  # 모델
    param_grid=rf_param_grid,  # 하이퍼파라미터 그리드
    cv=5,  # 교차 검증 횟수
    scoring='accuracy'  # 평가 지표
)

# 그리드 서치 실행 및 최적 파라미터 찾기
rf_grid_search.fit(X_train_encoded, train_y)

# 최적의 파라미터 및 교차 검증 점수 출력
print("Random Forest:")
print(f"Best parameters: {rf_grid_search.best_params_}")
print(f"Best cross-validation score: {rf_grid_search.best_score_:.2f}")

# 소프트 투표 분류기 정의 및 훈련
voting_clf_soft = VotingClassifier(
    estimators=[('ada', ada_boosting), ('rf', random_forest), ('gb', gradient_boosting)],
    voting='soft'  # 소프트 투표 방식
)
voting_clf_soft.fit(X_train_encoded, train_y)

# 테스트 데이터에 대한 예측 확률 계산
pred_soft = voting_clf_soft.predict_proba(X_test_encoded)

# 클래스 라벨 출력
display(voting_clf_soft.classes_)

# 예측 확률 출력
display(pred_soft)

# 샘플 제출 파일 로드
sample_submission = pd.read_csv('sample_submission.csv')

# 예측 확률을 'Click' 열에 추가
sample_submission['Click'] = pred_soft[:, 1]

# 제출 파일 확인
sample_submission

# 제출 파일 저장
sample_submission.to_csv('baseline_submission_soft.csv', index=False)
