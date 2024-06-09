import torch
import torch.nn as nn
import torch.optim as optim
import load
import matplotlib.pyplot as plt
import model as M


device = torch.device("cuda")


# model = nn.Sequential(
#     nn.Conv2d(3, 4, 3, 1, 1),
#     nn.BatchNorm2d(4),
#     nn.LeakyReLU(True),
#     nn.Conv2d(4, 6, 3, 2, 1),
#     nn.LeakyReLU(True),
#     nn.Conv2d(6, 8, 3, 2, 1),
#     nn.LeakyReLU(True),
#     nn.Conv2d(8, 10, 3, 2, 1),
#     nn.LeakyReLU(True),
#     nn.BatchNorm2d(10),
#     nn.Conv2d(10, 12, 3, 2, 1),
#     nn.LeakyReLU(True),
#     nn.Conv2d(12, 14, 3, 2, 1),
#     nn.LeakyReLU(True),
#     nn.AdaptiveMaxPool2d(1),
#     nn.Flatten(),
#     nn.Linear(14, 26),
#     nn.Sigmoid()
# ).to(device)

model = nn.Sequential(
    nn.Conv2d(3, 4, 3, 1, 1),
    nn.BatchNorm2d(4),
    nn.LeakyReLU(True),
    M.ResBlock(4, 6, 2),
    M.ResBlock(6, 8, 2),
    M.ResBlock(8, 10, 2),
    M.ResBlock(10, 12, 2),
    M.ResBlock(12, 14, 2),
    nn.AdaptiveMaxPool2d(1),
    nn.Flatten(),
    nn.Linear(14, 26),
    nn.Sigmoid()
).to(device)


loss_func = nn.BCELoss()
opt = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-5)

loader = load.loader('train', device, batch_size=256)

for epoch in range(5):
    for i, batch in enumerate(loader):
        x, y = batch
        y_hat = model(x)

        opt.zero_grad()
        loss = loss_func(y_hat, y)
        loss.backward()
        opt.step()

        acc = torch.mean(((y_hat > 0.5) == y).float())

        print("epoch: {} batch: {}, loss: {}, acc: {}".format(epoch, i, loss, acc))

        # if i > 20:
        #     break


loader = load.loader('test', device, batch_size=256)

model.eval()
for batch in loader:
    x, y = batch
    y_hat = model(x)

    # opt.zero_grad()
    loss = loss_func(y_hat, y)
    # loss.backward()
    # opt.step()

    acc = torch.mean(((y_hat > 0.5) == y).float())

    print("eval, loss: {}, acc: {}".format(loss, acc))


y = ((y_hat > 0.5).int().detach().cpu().numpy()).reshape([-1])
for i in range(26):
    print("{}: {}".format(load.attr[i], y[i]))


img = ((x[0].detach().cpu().numpy() + 1) / 2 * 255).transpose([1, 2, 0]).astype(int)
# print(img)
plt.imshow(img)
plt.show()

model.cpu()
torch.save(model.state_dict(), r"./reid.pth")

