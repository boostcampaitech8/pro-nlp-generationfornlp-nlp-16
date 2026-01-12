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
alpha (float or list) : 데이터 불균형을 조절하는 가중치 (기본 None)
                -특정 클래스의 중요도를 높힐때 사용함 (소수데이터)
                -float: 모든 클래스에 동일 가중치
                -list: 클래스별 가중치 [1번, 2번, 3번, 4번, 5번]
ignore_index (int) : loss 계산에서 무시할 라벨 값 (기본 -100, HuggingFace 패딩 표준)
reduction(str) : 계산된 Loss를 어떻게 합칠지 결정함 ("mean", "sum", "none")
                
"""
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # alpha를 텐서로 변환 (클래스별 가중치 지원)
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha

    """
    Focal Loss Calculation 함수

    Args :
    inputs : 모델이 예측 한 값 (Logits) [Batch_size, Num_classes]
    targets : 정답 라벨 (Labels) [Batch_size]

    """
    def forward(self, inputs, targets):
        
        #0. 유효한 토큰 마스크 (ignore_index=-100인 패딩 제외)
        valid_mask = (targets != self.ignore_index)
        
        # 유효한 토큰이 없으면 0 반환
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # 유효한 토큰만 선택
        valid_inputs = inputs[valid_mask]
        valid_targets = targets[valid_mask]
        
        # 1. Log Softmax 계산 (수치 안정성 확보)
        log_probs = F.log_softmax(valid_inputs, dim=-1)

        # 정답 클래스에 해당하는 log_prob만 추출
        # gather를 위해 차원을 맞춤: [N] -> [N, 1]
        log_pt = log_probs.gather(-1, valid_targets.unsqueeze(-1)).squeeze(-1)

        #2. pt(확률) 계산
        #Cross Entropy는 -log(p)형식이라서, exp(-ce_loss)를 하면 원래 확률(p)이 나옴
        #pt: 모델이 정답을 맞출 확률 (0~1)
        pt = log_pt.exp()
        
        #3. Focal loss 공식 적용
        # (1-pt) : 모델이 틀릴 확률 (쉽게 맞추면 이게 0에 가까워진다)
        # ** self.gamma : 틀릴 확률에 제곱을 해서 쉬운 문제의 Loss를 0으로 확 깎아버린다.
        ce_loss = -log_pt
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha 적용 (클래스별 가중치)
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # 디바이스 맞추기
                alpha = self.alpha.to(valid_inputs.device)
                # 각 샘플의 클래스에 해당하는 alpha 선택
                alpha_t = alpha[valid_targets]
            else:
                alpha_t = self.alpha
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * ce_loss

        #4. 결과 반환(avg or sum)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
