# distance-preserving-encoder

이 모델은 **고차원 임베딩 벡터를 더 저차원(latent)으로 압축**할 때 **원본 임베딩 간의 거리적 구조**(KNN 이웃 관계, 거리 순위 등)를 잠재공간에서도 **최대한 보존**하는 인코더를 만드는 것을 목표로 합니다.

## Loss Function

이 모델은 목표에 맞춰 Loss Function을 제작하여 사용했습니다.

```python
total = (lambda_rank * rank_loss) + (lambda_pairdist * pairdist_loss)
```

#### 1. 순위 기반 손실 (RankLoss)
- 원본 임베딩의 **KNN 이웃 순위**를 잠재공간(z)에서도 동일하게 맞추도록 학습합니다.

#### 2. 쌍별 거리 차이 손실 (PairwiseDistanceLoss)
- 각 벡터 끼리
    - 원본 공간(x)에서의 유클리드 거리와
    - 잠재공간(z)에서의 유클리드 거리가
    유사하도록 학습합니다.


## SOTA Hyperparameter in this project

<details>
<summary>KNN Recall@5 = 0.82</summary>
<div markdown="1">

`text-embedding-3-small` 임베딩 1,000개에 대해 학습/테스트 (2024.07 기준)
```python
#train
hidden_dims = [768]
latent_dim = 512
k = 5
lambda_rank = 1.0
lambda_pairdist = 0.3
lr = 1e-3
epochs = 1000

#test
k = 5
```

</div>
</details>

