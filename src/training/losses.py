"""
Focal Loss Implementation
Focal Loss를 구현한 코드입니다.
바로 Train에 적용하기보단, 따로 적용하여 끼워넣는게 좋을 것 같아 따로 작성하였습니다.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Focal Loss Init 함수

Args:
gamma (float) : '어려운 문제'에 얼마나 더 집중할지 결정하는 값(기본(국룰)2.0)
                -얘가 클수록(>1) 쉬운 문제는 무시하고, 어려운 문제의 Loss비중이 커진다.
alpha (float) : 데이터 불균형을 조절하는 가중치 (기본 1.0)
                -특정 클래스의 중요도를 높힐때 사용함 (소수데이터)
reduction(str) : 계산된 Loss를 어떻게 합칠지 결정함 ("mean", "sum", "none")
                
"""
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction


    """
    Focal Loss Calculation 함수

    Args :
    inputs : 모델이 예측 한 값 (Logits) [Batch_size, Num_classes]
    targets : 정답 라벨 (Labels) [Batch_size]

    """
    def forward(self, inputs, targets):
        
        #1. 기본 Cross Entropy Loss 계산 (reducion = none 으로 설정하여 각 샘플별 Loss를 따로 구함)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        #2. pt(확률) 계산
        #Cross Entropy는 -log(p)형식이라서, exp(-ce_loss)를 하면 원래 확률(p)이 나옴
        #pt: 모델이 정답을 맞출 확률 (0~1)
        pt = torch.exp(-ce_loss)
        #3. Focal loss 공식 적용
        # (1-pt) : 모델이 틀릴 확률 (쉽게 맞추면 이게 0에 가까워진다)
        # ** self.gamma : 틀릴 확률에 제곱을 해서 쉬운 문제의 Loss를 0으로 확 깎아버린다.
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        #4. 결과 반환(avg or sum)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
