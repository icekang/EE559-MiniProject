import importlib
import sys
import torch

sys.path.append('.\Proj_341752_337188_250222')

project_number = 2
Model = importlib.import_module(f"Miniproject_{project_number}.model").Model
model = Model()

x = torch.rand(1, 3, 512, 512) * 255
out = model.predict(x)
# state_dict = model.model.state_dict()
# torch.save(state_dict, './bestmodel2.pth')
model = Model()
model.load_pretrained_model()
actual = model.predict(x)
print(torch.allclose(actual, out))