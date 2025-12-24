"""

Train.py에 이식하기 위해 trian.py를 상속받아 만든 py입니다.
바로 Train에 적용하면 충돌이 불가피하고 코드가 길어질 수 있다고 생각하였습니다.
혼란이 생길 수 있기에 부품 갈아 끼듯이 쓸 수 있게 따로 작성하였습니다.

"""


from trl import SFTTrainer
from .losses import FocalLoss

"""
학습단계 (Step) 마다 Loss를 계산하는 함수 오버라이딩

Args :
Model : 학습 중인 모델 객체
inputs : 모델에 들어갈 데이터(inputs_ids, attention_mask, lavbels 등)
return_outputs : loss 외에 모델 출력값도 반환할지 여부
"""
class FocalLossTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. 데이터 꺼내기
        labels = inputs.get("labels") # 정답지
        outputs = model(**inputs) # 모델 예측 수행
        logits = outputs.get("logits") # 예측값

        #2. Focal Loss 객체 생성 (필요하면 config에서 값을 받아오게끔 수정 가능함)
        # 클래스 불균형 가중치: 1번(755개), 2번(350개), 3번(259개), 4번(232개), 5번(28개)
        # 5번이 매우 적으므로 높은 가중치 부여
        # 추 후 맞춰지면 수정해야함.
        loss_fct = FocalLoss(
            gamma=2.0,
            alpha=[1.0, 2.2, 2.9, 3.3, 27.0],  # 클래스별 가중치
            ignore_index=-100,  # 패딩 토큰 무시
        )

        #3. Casual LM(Gemma, Llam 등)을 위한 shift연산
        #LLM은 "오늘 점심은" 을 보고나서 "제육볶음"을 맞추는 방식
        #즉, n번쨰 입력의 정답값은 n+1에 있다.
        shift_logits = logits[..., :-1, :].contiguous() #마지막 토큰의 예측값은 정닶이 없으니까 버림 (n+1에 정답)
        shift_labels = labels[..., 1:].contiguous()#첫 번째 토큰은 예측 대상이 아니니까 버림 (n번째가 입력이라고 했으니까)

        #4. 차원 평탄화 및 Loss 계산
        # view(-1,...) 평탄화 해서 한줄로 쭉펴서 (Batch * Sequnce, vocab) 형태로 만듦
        # 이래야 Loss 함수에 넣을 수 있다.
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss
    