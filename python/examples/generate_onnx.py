#!/usr/bin/env python3

import torch


class SimpleMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.gelu = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.input_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        gelu = self.gelu(hidden)
        output = self.fc2(gelu)
        output = self.sigmoid(output)
        return output


if __name__ == '__main__':

    with torch.no_grad():
        batch_size = 5
        seq_len = 8
        input_size = 4
        hidden_size = 6
        input = torch.randn(batch_size, seq_len, input_size).cuda()
        model = SimpleMLP(input_size, hidden_size).cuda()
        output = model(input)
        torch.onnx.export(model, input, "naive_model.onnx", verbose=True,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={
                              'input': {0: 'batch'}, 'output': {0: 'batch'}})
