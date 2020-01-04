from __future__ import print_function
import torch
import numpy as np

x = torch.empty(5, 3)
# print(x)

x = torch.rand(5, 3)
# print(x)

x = torch.zeros(5, 3, dtype=torch.long)
# print(x)


x = torch.tensor([5.5, 3])
# print(x)

x = x.new_ones(5, 3, dtype=torch.double)
# print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

print(x.size())

y = torch.rand(5, 3)
# print(x + y)

y.add_(x)
# print(y)

# print(x[:, 1])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
# print(x.size(), y.size(), z.size())


x = torch.randn(1)
# print(x)
# print(x.item())

a = torch.ones(5)
# print(a)

b = a.numpy()
# print(b)

a.add_(1)
# print(a)
# print(b)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
# print(a)
# print(b)


# gradients
x = torch.ones(2, 2, requires_grad=True)
# print(x)

y = x +2
# print(y)

# print(y.grad_fn)

z = y*y*3
out = z.mean()

# print(z, out)

a = torch.randn(2,2)
a = ((a * 3) / (a - 1))
# print(a.requires_grad)
a.requires_grad_(True)
# print(a.requires_grad)
b = (a * a).sum()
# print(b.grad_fn)


out.backward()
print(x)
print(out)
print(x.grad) # becouse our function from which we are computing gradient is out=z.mean(); z=y*y*3; y=x+2


x = torch.randn(3,requires_grad=True)

y = x*2
while y.data.norm() < 1000:   #we are resizing y while norm of y will be less than 1000
    y = y * 2

print(y)
print(x.grad)
print(y.grad_fn)
v = torch.tensor([0.1,1.0,0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())