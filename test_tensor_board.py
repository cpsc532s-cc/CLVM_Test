from tensorboardX import SummaryWriter

writer = SummaryWriter()

for i in range(10):
    writer.add_scalar('data/scalar1', i, i)

writer.close()
