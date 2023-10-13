from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("logs")#命名存储tensorboard的日志文件夹名称，会在当前项目文件夹下生成一个名为logs的日志文件夹
#y=x
for i in range(100):
    writer.add_scalar("y=x",i,i)
writer.close()#本句一定要有，不然会报错
