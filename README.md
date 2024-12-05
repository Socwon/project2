<img src="https://github.com/user-attachments/assets/21a18d01-0883-4299-a247-909fed87ab4c" width="700" height="400"/>

## MobileBERT를 활용한 로블록스 리뷰 분석 프로젝트  
<div>
<img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white"/> 
<img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white"/> 
<img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white"/>
</div>

## 1. 개 요 
### 1.1 로블록스의 영향력
로블록스는 사용자 생성 콘텐츠(UGC), 몰입형 소셜 경험, 혁신적인 수익화 전략, 크로스 플랫폼 접근성 등으로 게임 산업에 큰 영향을 미쳤다. 게임 개발의 민주화를 이끌었고, 소셜 상호작용과 협력을 강조하여 다른 게임들도 이를 채택하게 했다. 가상 아이템 거래와 경제 시스템을 통해 개발자와 플레이어 모두에게 수익 창출 기회를 제공하며, 다양한 기기에서의 접근성을 통해 크로스 플랫폼 플레이의 중요성을 강조했다. 또한, 교육적 콘텐츠와 인플루언서의 참여를 통해 게임 트렌드를 형성하고, 젊은 개발자들에게 창작의 기회를 제공하여 게임 산업의 변화를 이끌었다. 


(출처 : https://www.blog.zeusx.com/post/roblox-s-influence-on-game-trends-shaping-the-future-of-gaming

### 1.2 문제 정의
MobileBERT 모델을 사용하여 로블록스의 리뷰 데이터를 자동으로 분석하고, 긍정적 및 부정적 평가를 분석한다. 이를 통해 로블록스 개발자 시점이 되어 사용자 피드백을 효율적으로 파악하고, 서비스 개선 방법을 알아본다.

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
'date' 컬럼을 datetime 형식으로 변환하여, 연도별로 데이터를 그룹화할 수 있도록 했다. 또한 'date' 컬럼에서 연도만 추출하여, 연도별 리뷰 데이터를 2023년과 2024년으로 나누어 분석할 수 있도록 했다.
그 후 date 컬럼에서 연도만 추출하여 year 라는 새로운 컬럼을 추가해 그 기준으로 데이터를 그룹화하고, 연도별 리뷰 수와 평균 평점 등을 계산할 수 있었다.
리뷰의 평점이 1에서 2일 경우는 부정적인 리뷰로 4에서 5일 경우는 긍정적인 리뷰로 라벨링하고 평점이 3인 경우는 중립적인 리뷰로 분류하였다. 
이후 중립 리뷰를 제외하여 부정적 또는 긍정적인 리뷰만을 분석 대상으로 삼았다.


중립 리뷰를 제외한 데이터 수

<img src="https://github.com/user-attachments/assets/4bc01321-5e6e-400d-bd3e-102aa907dd86" width="700" height="400"/>

긍정적인 리뷰가 부정적인 리뷰보다 더 많았다. 이는 사용자들이 대체로 긍정적인 경험을 공유하고 있음을 시사한다.



## 3. 학습 데이터 구축
#### 패키지
<div>
<img src="https://img.shields.io/badge/transformers-409FFF?style=flat-square&logo=transformers&logoColor=white"/>
<img src="https://img.shields.io/badge/torch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/numpy-013243?style=flat-square&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit-learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
</div>
<br>

### 3.1 학습 데이터와 분석 데이터
| artist | summary | text  | 
|-------|-----|-------|
|Unwritten Law|Genocide| Well who are you what do you know And I guess it only goes to show Don't think you're ready for the fall Do you think you really wanta play.. |
| Tori Amos | Butterfly | Stinky soul get a little lost in my own Hey General, need a little love in that hole of yours One ways, now, and..|
| .. | .. | ... |
| John Denver | Downhill Stuff | Blue river blues I'd rather be outside Here I am inside Watching it rain Blue river blues I'd rather be somewhere.. | 
| Judy Garland | Why Was I Born? | Why was I born? Why am I livin'? What do I get? What am I givin'? Why do I want a thing I.. | 

데이터셋을 학습 데이터와 분석 데이터로 분할하였다.
원본 데이터셋의 link열은 삭제하였으며 song열의 이름을 summary로 변경하고 모델 학습의 효율성을 높이기 위해 학습 데이터는 3000개의 샘플을, 분석 데이터는 5000개의 샘플을 랜덤으로 추출하여 새로운 데이터셋을 만들었다.

### 3.2 모델 학습
각 가사를 입력으로 받고 해당하는 제목을 목표 출력으로 설정하여 이 모델을 노래 제목 예측에 맞게 미세 조정을 수행하여 학습했다.

## 4. T5 학습 결과
### 4.1 결과
<img src="https://github.com/Socwon/Spotify_Summary/assets/101037584/e3904690-c279-4f40-8073-24a27081a274"/>

사전 훈련된 T5 모델을 사용하여 노래 가사를 기반으로 제목을 생성하도록 미세 조정을 수행했다.

<img src="https://github.com/Socwon/Spotify_Summary/assets/101037584/f63abc24-fa1e-4ad4-abb2-1a51eaad9e73" width="800" height="250"/>
<img src="https://github.com/Socwon/Spotify_Summary/assets/101037584/13279650-a668-491b-b9a6-99e6dbb8c95e" width="800" height="250"/>

<img src="https://github.com/Socwon/Spotify_Summary/assets/101037584/721ad448-900b-42dd-a9f8-b1aa3e7eb200" width="330" height="500"/>


### 4.2 모델 평가
<img src="https://github.com/Socwon/Spotify_Summary/assets/101037584/af3d0315-2796-4d18-bd32-8e025d8f8f34" width="700" height="400"/>

모델의 성능을 평가하기 위해 제목 길이 분포 히스토그램을 생성하였다. 모델이 생성한 제목의 단어 수 분포를 나타내는데 x축은 제목의 길이, y축은 해당 길이의 제목이 생성된 빈도수이다.
대부분의 제목이 5~10 단어로 구성되어 있음을 알 수 있다. 그러나 특정 길이에서 제목의 빈도가 높다는 것은 모델이 특정 패턴의 제목을 선호한다는 것을 볼 수 있다.
모델의 학습 손실은 2.3077, 평가 손실은 1.6959로 나타났고, 특히 ROUGE-1과 ROUGE-L에서 높은 점수를 기록하여 단어 단위 및 문맥적인 유사성을 잘 유지하고 있다. 평균 생성된 제목 길이는 5.6119 단어로, 모델이 간결하고 적절한 길이의 제목을 생성하고 있음을 알 수 있다.
학습 및 평가 시간 또한 효율적으로 관리되었다. 학습 중 처리 속도와 평가 중 처리 속도의 차이는 예상된 결과로 모델의 학습과 예측 성능이 안정적임을 보여주었다.

## 5. 느낀점 및 배운점

이번 프로젝트를 통해 다양한 측면에서 많은 것을 배울 수 있었다. T5 모델을 활용하여 노래 가사로부터 제목을 예측하는 작업은 매우 흥미로운 도전이었다. 이 작업을 통해 데이터의 품질과 양이 모델 성능에 큰 영향을 미친다는 점을 다시 한 번 깨닫게 되었다. 초기에는 3000개의 샘플로 학습을 진행하였으나 이는 모델의 일반화 능력을 향상시키기에는 다소 부족하다 생각하여 이후 5000개의 샘플을 분석하면서 더 많은 데이터를 통해 모델이 보다 풍부한 패턴을 학습할 수 있음을 경험하여 데이터의 중요성을 다시 한 번 상기시키는 계기가 되었다. ROUGE 지표는 문장 생성 모델의 전반적인 성능을 평가하기에는 제한적이여서 생성된 제목의 질을 더 정확하게 평가할 수 있는 방법을 모색하지 않은 점이 아쉽습니다. 하이퍼파라미터 튜닝은 모델 성능을 최적화하는 데 중요한 역할을 하며, 작은 변화가 모델의 예측 정확도에 큰 영향을 미칠 수 있음을 배웠다. 이를 통해 모델을 최적화하기 위한 지속적인 실험과 조정의 중요성도 깨달았다. 
또 프로젝트 진행 중 다양한 문제에 직면하게 되었고, 그중 코드의 효율성을 높이기 위해 tqdm 라이브러리를 사용하여 진행 상황을 모니터링하고 생성된 결과를 한눈에 보기 쉽게 출력하는 방법을 적용해보며 여러 방법을 시도하면서 문제 해결 능력이 향상되었다. 이 경험은 실제 프로젝트 환경에서 문제를 해결하는 데에도 큰 도움이 될 것 같다.
