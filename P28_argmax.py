import torch

outputs=torch.tensor([[0.1,0.2],
                      [0.05,0.4]])
print(outputs.argmax(1))#为1表示求横向tensor数据中的最大值的下标，分别是0.1<0.2,0.05<0.4。0.2和0.4下标为1
print(outputs.argmax(0))#为0表示求纵向tensor数据中的最大值的下标，分别是0.1>0.05,0.2<0.4。


preds=outputs.argmax(1)
targets=torch.tensor([0,1])
print((preds==targets).sum())
print((preds==targets).sum()/2)