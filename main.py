from neurality import NeuralNet

nn = NeuralNet(8, 8, 64, 64**1.25)
print('Created')

for i in range(1, 10 + 1):
    nn.cycle()
    print(f'cycle {i}')