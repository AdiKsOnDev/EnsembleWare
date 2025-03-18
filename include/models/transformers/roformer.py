from include.models.basemodel import BaseModel


class RoformerModel(BaseModel):
    def __init__(self, model_name, num_labels=2):
        super().__init__(model_name, num_labels)
