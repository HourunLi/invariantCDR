x1 =  [ 0.1845, -1.9534,  0.3681, -0.5433, -0.6816, -0.8846, -0.2946, -0.7390]
x2 =  [-0.6610,  0.8877,  0.9336, -2.2997,  0.5129,  0.9605,  0.1560, -0.8687]
res = 0
for i in range(len(x1)):
    res += x1[i]*x2[i]
print(res)