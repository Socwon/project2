# preprocess.py

import pandas as pd
import matplotlib.pyplot as plt

def preprocess_reviews(review_df):
    print("데이터 전처리를 시작합니다...")  # 진행 상황 출력

    # 라벨링 기준
    def label_review(score):
        if score == 1 or score == 2:
            return 0  # 부정
        elif score == 3:
            return None  # 중립 (이후 필터링)
        elif score == 4 or score == 5:
            return 1  # 긍정

    # 'label' 컬럼 추가
    review_df['label'] = review_df['score'].apply(label_review)
    print(f"총 {len(review_df)}개의 리뷰에 라벨이 추가되었습니다.")

    # 중립 리뷰는 제외
    review_df = review_df.dropna(subset=['label'])
    print(f"중립 리뷰를 제외한 데이터 수: {len(review_df)}")

    # 히스토그램 시각화
    plt.figure(figsize=(8, 6))
    plt.hist(review_df['score'], bins=5, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.show()  # 그래프 출력

    # 긍정/부정 리뷰에서 샘플링
    positive_reviews = review_df[review_df['label'] == 1]
    negative_reviews = review_df[review_df['label'] == 0]

    positive_sample = positive_reviews.sample(n=1000, random_state=42)
    negative_sample = negative_reviews.sample(n=1000, random_state=42)

    # 샘플 합치기
    sampled_reviews = pd.concat([positive_sample, negative_sample])

    # 샘플링된 데이터프레임 확인
    print(f"샘플링된 긍정/부정 리뷰의 수: {len(sampled_reviews)}")

    # 샘플링된 데이터 저장 (예: CSV)
    sampled_reviews.to_csv('sampled_reviews.csv', index=False)
    print("샘플링된 리뷰가 'sampled_reviews.csv'에 저장되었습니다.")

    return sampled_reviews

# 데이터 파일 읽기 및 처리 예시
if __name__ == "__main__":
    # 'reviews_2023_2024.json' 파일을 읽어들이고, 전처리 실행
    review_df = pd.read_json('reviews_2023_2024.json', orient='records', encoding='utf-8')

    # 전처리 함수 실행
    sampled_reviews = preprocess_reviews(review_df)
