""""""""""""""""""""""""""""""""""

Train.py에 이식하기 위해 trian.py를 상속받아 만든 py입니다.
바로 Train에 적용하면 충돌이 불가피하고 코드가 길어질 수 있다고 생각하였습니다.
혼란이 생길 수 있기에 부품 갈아 끼듯이 쓸 수 있게 따로 작성하였습니다.

""""""""""""""""""""""""""""""""""


from trl import SFTTrainer
from .losses import FocalLoss

class FocalLossTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        ouputs = model(**inputs)
        logits = outputs.get("logits")

        #Focal Loss 호출 (gamma값 조절 가능)
        loss_fct = FocalLoss(gamma=2.0)

        #Next Token Prediction을 위한 shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        #Loss 계산
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss