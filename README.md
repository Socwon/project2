<img src="https://github.com/Socwon/Spotify_Summary/assets/101037584/13b559d7-7593-4f7b-832e-623fb1ee88b1" width="700" height="200"/>

## MobileBERT를 활용한 로블록스 리뷰 분석 프로젝트  
<div>
<img src="https://img.shields.io/badge/python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white"/> 
<img src="https://img.shields.io/badge/pytorch-%23EE4C2C.svg?&style=for-the-badge&logo=pytorch&logoColor=white"/> 
<img src="https://img.shields.io/badge/pycharm-%23000000.svg?&style=for-the-badge&logo=pycharm&logoColor=white"/>
</div>

## 1. 개 요 
### 1.1 노래 제목의 영향력
노래 제목은 노래의 핵심을 나타내며, 듣는 이들에게 노래의 주제나 감정을 전달하는 역할을 한다. 노래 제목은 노래를 소개하고, 청취자의 관심을 끌고, 노래의 주제나 감정을 전달하는 역할을 한다. 대중적인 장르의 경우 노래의 의미가 명확해지는 것이 중요하며, 노래 제목은 노래 전체에서 가장 기억에 남는 부분이어야 하므로 명확성이 중요하다. 이는 노래의 인기와 성공에 영향을 미치는 요소 중 하나이며 또한 마케팅적으로도 중요하고, 청취자의 호기심을 자극하여 노래를 들을 가능성을 높일 수 있다. 따라서 노래 제목을 의도적으로 선택하는 것이 중요하다.

(출처 : https://soundfly.com/courses/intro-to-scoring-for-film-and-tv?utm_source=Flypaper&utm_campaign=Flypaper-ad-system&utm_medium=sidebar-ad)

### 1.2 문제 정의
Spotify Million Song Dataset는 스포티파이라는 음악 앱의 노래 이름, 아티스트 이름, 노래 링크 및 가사가 포함 되어 있는 데이터셋이다. 노래 추천, 노래 분류 등에 사용할 수 있으며 이 프로젝트에서는 대부분의 사람들이 즐겨 듣는 노래는 어떤 내용의 가사가 담겨져 있으며 이에 따른 제목이 무엇인지 예측하는 인공지능 모델을 개발하고자 한다.

## 2. 데이터
### 2.1 원시 데이터
[Spotify Million Song 데이터셋](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset)<br/>
[Spotify 홈페이지](https://open.spotify.com/)

- 데이터명

| artist | song | link  | text  |
|-------|-----|-------|-------|
|아티스트 이름|노래 제목| 노래 링크 | 노래 가사 |

- 활용할 데이터 예시

| artist | song | link  | text  |
|-------|-----|-------|-------|
|Alison Krauss|Maybe| /a/alison+krauss/maybe_20006156.html | Yesterday the odds were stacked In favor of my expectations Flyin' above the rest, Never fa... |
| Aerosmith | The Hand That Feeds | /z/zz+top/36+22+36_20149356.html  | Doctor, doctor, doctor Please, doctor, doctor, please Doctor, doctor, doctor Feel like a ol...  |
| .. | .. | ...  | ...  |
| ZZ Top | 36-22-36 | /b/barbra+streisand/i+wont+last+a+day+without+you_20699679.html  | What, what, what you want? Hey My thing is a real fine thing It's a thing, it's a real fine th...  |
| ZZ Top | 2000 Blues | /z/zz+top/2000+blues_10198797.html  | A hundred thousand dollars Wouldn't touch the price I paid Of the hundred thousand moments ...  |

데이터는 총 57494건이 있다.


### 2.2 탐색적 데이터
- 노래를 가장 많이 낸 아티스트 15명과 노래를 가장 적게 낸 아티스트 15명
<img src="https://github.com/Socwon/Spotify_Summary/assets/101037584/d45fb3e6-8d97-429c-b428-2c11cf36686d" width="700" height="400"/>

제일 노래를 많이 낸 아티스트는 'Donna Summer'이며 노래는 191개이고 제일 노래를 적게 낸 아티스트는 5명이며 노래는 1개씩 냈다.

- 불용어를 제외한 모든 노래 가사에서 자주 등장하는 키워드 5가지
<img src="https://github.com/Socwon/Spotify_Summary/assets/101037584/98205e6b-d976-4dbd-9d43-9db8ab37a7ed" width="450" height="400"/>

제일 많이 등장한 키워드는 순서대로 "Love", "Time", "One", "Heart", "I'm" 이다.
따라서 사랑, 시간에 관련된 주제의 노래가 많이 발표된다는 것을 알 수 있다.

### 2.3 추출한 데이터에 대한 탐색적 데이터 분석
- 모든 노래의 평균 가사 길이
<img src="https://github.com/Socwon/Spotify_Summary/assets/101037584/318475fc-1f52-4102-ba87-03f0a6cfdce3"/>

- 모든 노래의 가사 길이 시각화
<img src="https://github.com/Socwon/Spotify_Summary/assets/101037584/a0cc12b0-4f2f-48e3-aad2-36b7a8677f26" width="700" height="400"/>

평균적으로 100글자에서 300글자 사이의 가사를 가진 노래가 가장 많이 나온다는 것은 음악을 구성하는 가사의 적절한 길이가 중요하다는 것을 보여준다. 따라서 노래 제작자들은 이러한 범위에 가사를 맞추는 것이 노래의 인기와 성공에 도움이 될 수 있다는 것을 고려하며 노래를 제작한다는 것을 알 수 있다.


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
