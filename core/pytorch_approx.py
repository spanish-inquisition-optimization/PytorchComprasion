import torch
from core.utils import ApproxDump


def polynom(a, x):
    return torch.matmul(x.unsqueeze(-1).pow(torch.arange(0, a.size()[0])), a)


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

    return ApproxDump('pytorch', x.numpy(), w.detach().numpy(), loss_history)
