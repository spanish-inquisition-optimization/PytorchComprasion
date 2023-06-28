import torch
import math
from matplotlib import pyplot as plt


def polynom(a, x):
    return torch.matmul(x.unsqueeze(-1).pow(torch.arange(0, a.size()[0])), a)


class ApproxDump:
    def __init__(self, x, y, w, loss_history):
        self.x = x
        self.y = y
        self.w = w
        self.loss_history = loss_history

    def visualize(self):
        plt.plot(self.loss_history, label='loss')
        plt.grid()
        plt.legend()
        plt.yscale('log')
        plt.show()

        plt.plot(self.x.numpy(), self.y.numpy(), label='actual')
        plt.plot(self.x.numpy(), polynom(self.w, self.x).detach().numpy(), label='predicted')
        plt.legend()
        plt.show()


def polynom_approx(x, y, deg, steps, optimizer_supplier, scheduler_supplier=None):
    loss_history = []
    w = torch.randn(deg, requires_grad=True)
    optimizer = optimizer_supplier([w])
    scheduler = scheduler_supplier(optimizer) if scheduler_supplier is not None else None

    for t in range(0, steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(polynom(w, x), y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(loss)

        loss_history.append(float(loss))

    return ApproxDump(x, y, w, loss_history)
