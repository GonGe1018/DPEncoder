# distance-preservation-encoder

이 모델은 **고차원 임베딩 벡터를 더 저차원(latent)으로 압축**할 때 **원본 임베딩 간의 거리적 구조**(KNN 이웃 관계, 거리 순위 등)를 잠재공간에서도 **최대한 보존**하는 인코더를 만드는 것을 목표로 합니다.

### SOTA Hyperparameter in this project
---
####  `KNN Recall@5 = 0.82`
OpenAI `text-embedding-3-small` 임베딩 1,000개에 대해 학습/테스트 (2024.07 기준)
```python
#train
hidden_dim = [768]
latent_dim = 512
k = 5
lambda_rank = 1.0
lambda_pairdist = 0.3
lr = 1e-3
epochs = 1000

#test
k = 5
```
---