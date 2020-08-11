# Machine Learning Notebook
기계학습에 대해서 이론부터 실제 코드구현까지 정리한 repository 입니다.
실제 분석시 많이 사용되는 부분들을 이론및 예제까지 정리를 합니다. 
또한 기계학습 또는 딥러닝 연구원 또는 엔지니어로 면접시 도움이 될 만한 내용들을 정리하였습니다. 

> 현재 많은 contents들을 지속적으로 만들고 있습니다. 



## Contents

1. 000 ~ 099 : Statistics, Linear Algebra (이론)
2. 100 ~ 199 : Linear Regression & Basic Machine Learning
3. 200 ~ 299 : Useful Analysis & Machine Learning 
4. 300 ~ 399 : Machine Learning Algorithms
5. 400 ~ 499 : ..
6. 500 ~ 599 : Deep Learning Basics
7. 600 ~ 699 : Convolution Neural Networks
8. 700 ~ 799 : Recurrent Neural Network Algorithms



# 면접 질문과 답변 정리

해당 repository가 면접을 위해서 만들어진 노트북은 아니지만, 면접을 위해 필요한 사항들 또한 정리를 했습니다. 

## 통계 문제와 답변

1. [Performance Test 그리고 ROC, AUC](blob/master/001%20Performance%20Test%20(ROC%2C%20AUC%2C%20Confusion%20Matrix)/performance%20test.ipynb) 에 대해서 설명하고 수학적 공식을 쓰세요



## 기계학습 문제와 답변

1. Eigenvalue 그리고 eigenvector 이란 무엇인가? 왜 중요한가?
   - [Eigenvalue & Eigenvector 자세한 내용 참고](blob/master/170%20Eigenvalue%20and%20Eigenvector/Eigenvalue%20and%20Eigenvector.ipynb)
   - 선형변화 A를 했을때, 크기는 변하지만 방향은 변하지 않는 것이 eigenvector이고, 얼마만큼 크기가 변했는지를 나타내는 것은 eigenvalue. 이때 eigenvector는 null vector (영벡터)가 아니다.
2. PCA 알고리즘의 구현방법은?
   - 구현방법
     1. 데이터에 대해서 Standardization
     2. Covariance Matrix 계산
     3. Eigenvalue and eigenvector of the covariance matrix 계산
     4. Eigenvalue의 값에 따라서 eigenvalue 그리고 eigenvector를 정렬
     5. Dimensional reduction
   - [구현방법 참고](blob/master/210%20Principle%20Component%20Analysis%20(PCA)/Extracting%20PCA.ipynb)



## 딥러닝

