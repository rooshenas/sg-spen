

class Config:
  def __init__(self):
    self.dropout = 0.0
    self.layer_info = None
    self.en_layer_info = None
    self.weight_decay = 0.001
    self.l2_penalty = 0.0
    self.inf_penalty = 0.0
    self.en_variable_scope = "en"
    self.fx_variable_scope = "fx"
    self.spen_variable_scope = "spen"
    self.inf_rate = 0.1
    self.noise_rate = 0.2
    self.learning_rate = 0.001
    self.margin_weight = 100.0
    self.hidden_num = 300
    self.input_num = 0
    self.output_num = 0
    self.loglevel = 0

