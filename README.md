<img src="https://github.com/user-attachments/assets/21a18d01-0883-4299-a247-909fed87ab4c" width="700" height="400"/>

## MobileBERT를 활용한 로블록스 리뷰 분석 프로젝트  
<div>
<img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white"/> 
<img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white"/> 
<img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white"/>
</div>

## 1. 개 요 
### 1.1 프로젝트의 목적
이번 프로젝트의 주요 목표는 **로블록스(Roblox)**라는 인기 있는 온라인 게임에 대한 사용자들의 리뷰를 분석하는 것이다. 
로블록스는 전 세계적으로 수백만 명의 사용자가 즐기는 게임 플랫폼으로, 각종 사용자 생성 콘텐츠와 게임을 제공하여 큰 인기를 끌고 있다.
게임을 즐기는 사람들의 경험과 피드백은 게임의 발전에 중요한 영향을 미치기 때문에 이러한 리뷰를 분석하는 것은 매우 중요하다.
본인도 로블록스를 즐기는 사람으로서 나와 같은 사람들은 이 게임에 대해 어떻게 생각하는지와 어떤 경험을 공유하고 있는지를 긍정적 부정적으로 분류하여 파악할 것이다.
이를 통해 로블록스의 사용자 경험에 대한 인사이트를 얻고 게임 개발자나 관련 기업들이 게임 개선을 위한 방향성을 찾는 데에도 도움을 줄 수 있다.

### 1.2 프로젝트의 의미
본 프로젝트는 자연어 처리(NLP) 기술을 활용하여 로블록스 사용자들이 남긴 리뷰를 감성 분석하고, 각 리뷰가 긍정적인지 부정적인지 분류하는 방식으로 진행된다.
이는 게임의 사용자 경험을 정량화하고 게임 개발자들이 사용자 요구에 맞는 개선점을 발견하는 데 중요한 정보를 제공한다.

## 2. 데이터 분석
### 2.1 데이터 수집 방법 (웹 크롤링)
- **리뷰 사이트 접근** : 로블록스 웹사이트에서 리뷰 데이터를 포함하는 페이지에 접근하여, 각 리뷰의 텍스트 및 평점 정보를 추출하였습니다.

- **데이터 추출** : 페이지에서 필요한 정보를 추출하기 위해 BeautifulSoup을 사용하여 HTML 구조에서 리뷰 텍스트(text)와 평점(score)을 선택하고 Python 리스트 형태로 저장했다.

- **데이터 저장** : 추출된 데이터는 JSON 파일로 **reviews_2023_2024.json**라는 이름으로 저장하여 분석을 진행했다.


### 2.2 수집된 데이터의 형식
- **review** : 사용자 리뷰 텍스트
- **score** : 해당 리뷰의 평점 (1~5 사이의 값)
- **date** : 리뷰 작성 날짜

### 2.3 데이터 전처리
중립 리뷰를 제외한 데이터 수
<img src="https://github.com/user-attachments/assets/681c3839-105b-4d1b-85e0-cb45f628a20b" width="800" height="300"/>

중립 리뷰를 제외한 12는 부정, 45는 긍정 리뷰 데이터 수 일부 출력 

<img src="https://github.com/user-attachments/assets/4bc01321-5e6e-400d-bd3e-102aa907dd86" width="700" height="400"/>

'date' 컬럼을 datetime 형식으로 변환하여, 연도별로 데이터를 그룹화할 수 있도록 했다. 또한 'date' 컬럼에서 연도만 추출하여, 연도별 리뷰 데이터를 2023년과 2024년으로 나누어 분석할 수 있도록 했다.
그 후 date 컬럼에서 연도만 추출하여 year 라는 새로운 컬럼을 추가해 그 기준으로 데이터를 그룹화하고, 연도별 리뷰 수와 평균 평점 등을 계산할 수 있었다.
리뷰의 평점이 1에서 2일 경우는 부정적인 리뷰로 4에서 5일 경우는 긍정적인 리뷰로 라벨링하고 평점이 3인 경우는 중립적인 리뷰로 분류하였다. 
이후 중립 리뷰를 제외하여 부정적 또는 긍정적인 리뷰만을 분석 대상으로 삼았다.

따라서 긍정적인 리뷰가 부정적인 리뷰보다 더 많았다. 이는 사용자들이 대체로 긍정적인 경험을 공유하고 있음을 시사한다.


## 3. KoELECTRA 모델 학습 및 테스트
본 프로젝트에서는 KoELECTRA 모델을 활용하여 로블록스 리뷰 데이터를 학습했다. 모델은 긍정(1) 및 부정(0) 리뷰를 분류하는 이진 분류 문제로 정의되었으며, 아래와 같은 프로세스를 통해 학습과 평가를 진행했다.

### 3.1 모델 학습
- 학습 초기
<img src="https://github.com/user-attachments/assets/45f7b54b-9df4-439f-a5ff-da28aa9b3c3e">

- 학습 진행
<img src="https://github.com/user-attachments/assets/1e6bf09f-7f5d-47bb-a80e-2307961312a2">

- 최종
<img src="https://github.com/user-attachments/assets/820fa1a3-7ddb-4fae-9520-71fd4c14043e">


### 3.2 


## 4. 학습 결과
### 4.1 


### 4.2 모델 평가


## 5. 느낀점 및 배운점

