import torch

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(
            self.forward_hook
        )

        target_layer.register_full_backward_hook(
            self.backward_hook
        )

    def forward_hook(self, module, inp, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x):

        output = self.model(x)

        score = output.squeeze()

        self.model.zero_grad()

        score.backward()

        gradients = self.gradients[0]

        activations = self.activations[0]

        weights = gradients.mean(
            dim=(1,2)
        )

        cam = torch.zeros(
            activations.shape[1:],
            device=x.device
        )

        for i,w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)

        cam -= cam.min()

        cam /= (
            cam.max() + 1e-8
        )

        return cam.detach().cpu().numpy()