import torch

class FreezeParameters:
    def __init__(self, *modules):
        self.modules = modules
        self.parameters = []
        [self.parameters.extend(mod.parameters()) for mod in modules]
        self.original_states = [param.requires_grad for param in  self.parameters]

    def __enter__(self):
        for param in self.parameters:
            param.require_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(self.parameters):
            param.requires_grad = self.original_states[i]
