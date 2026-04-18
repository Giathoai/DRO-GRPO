import torch
import torch.nn.functional as F
from trl import GRPOTrainer

class RobustGRPOTrainer(GRPOTrainer):
    def __init__(self, dr_temperature=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dr_temperature = dr_temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        """Ghi đè hàm Loss để tính Trọng số Phân phối (DR Weights)"""
        outputs = super().compute_loss(model, inputs, return_outputs=True)
        base_loss = outputs[0] if return_outputs else outputs
        
        if len(base_loss.shape) > 0 and base_loss.shape[0] > 1:
            detached_loss = base_loss.detach()
            dr_weights = F.softmax(detached_loss / self.dr_temperature, dim=0)
            robust_loss = torch.sum(dr_weights * base_loss)
        else:
            robust_loss = torch.mean(base_loss)
            
        return (robust_loss, outputs[1]) if return_outputs else robust_loss