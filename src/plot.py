import matplotlib.pyplot as plt

plt.switch_backend("QT5Agg")

l = []
with open("output.txt", "r") as file:
    for i in file:
        l.append(float(i))

plt.plot(l)
plt.show()