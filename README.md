# distance-preservation-encoder



### SOTA hyperparameter in this project
opeani의 `text-embedding-3-small` 모델의 임베딩 데이터를 1000개 학습 시켰을 때 기준입니다.

#### 1st
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

#### 2nd
```python

```
