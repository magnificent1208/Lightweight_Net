'''
Note：
Knowledge distillation 知识蒸馏

prerequisite：
softmax
cross-entropy

Knowledge distillation：
    集成学习 Ensemble Learning
        多模型训练，最后加权使用。
    知识蒸馏的目的--模型压缩
        目的：
        已经有工作证明，在集成模型中的知识(knowledge)压缩到方便部署的单个模型是可行的。
        模型压缩方法比较：
            1. 模型压缩：在已经训练好的模型上进行压缩，使得网络带有更少的参数 (知识蒸馏的方向）
            2. 直接训练小型网络：从改变网络结构出发 (SqueezeNet，Mobilelnet)

    知识蒸馏的方法
    针对大型数据集的集成模型方法
    复现知识蒸馏网络
--------------------------------
论文 KD in NN

'''


from teacher import teach
from student import stu
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch.autograd import Variable
from utils import get_training_dataloader, get_test_dataloader
from torch.nn import functional as F

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
CIFAR10_TRAIN_MEAN = (0.485, 0.456, 0.406)
CIFAR10_TRAIN_STD = (0.229, 0.224, 0.225)

#total training epoches
cifar10_training_loader = get_training_dataloader(
    CIFAR10_TRAIN_MEAN,
    CIFAR10_TRAIN_STD,
)

cifar10_test_loader = get_test_dataloader(
    CIFAR10_TRAIN_MEAN,
    CIFAR10_TRAIN_STD,
)
EPOCH = 30
T,lambda_stu=5.0,0.05
#net = xception()
teacher=teach()
teacher.load_state_dict(torch.load("teacherNet.pth"))
teacher.eval()
teacher.train(mode=False)
student=stu()
optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
lossKD=nn.KLDivLoss()
lossCE= nn.CrossEntropyLoss()
def train(epoch):

    student.train()
    for batch_index, (images, labels) in enumerate(cifar10_training_loader):
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        images = Variable(images)
        labels = Variable(labels)
        optimizer.zero_grad()
        y_student = student(images)
        loss_student = lossCE(y_student, labels)
        y_teacher=teacher(images)
        loss_teacher=lossKD(F.log_softmax(y_student / T, dim=1),
                                  F.softmax(y_teacher / T, dim=1))
        loss=lambda_stu*loss_student+(1-lambda_stu)*T*T*loss_teacher
        loss.backward()
        correct_1=0
        _,pred=y_student.topk(5,1,largest=True,sorted=True)
        labels=labels.view(labels.size(0),-1).expand_as(pred)
        correct=pred.eq(labels).float()
        correct_1 += correct[:, :1].sum()
        optimizer.step()
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\ttop-1 accuracy: {:0.4f}\t'.format(
            loss.item(),
            (100.*correct_1)/len(y_student),
            epoch=epoch,
            trained_samples=batch_index * 128 + len(images),
            total_samples=len(cifar10_training_loader.dataset)
        ))
    scheduler.step()
def eval_training(epoch):
    student.eval()
    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar10_test_loader:
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        with torch.no_grad():
            images, labels = Variable(images), Variable(labels)

        outputs = student(images)
        loss = lossCE(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
    loss = test_loss / len(cifar10_test_loader.dataset)
    accuracy = (100. * correct.float()) / len(cifar10_test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar10_test_loader.dataset),
        (100.*correct.float())/ len(cifar10_test_loader.dataset)
    ))
    return loss,accuracy

def save_best(loss, accuracy, best_loss, best_acc):
    if best_loss == None:
        best_loss = loss
        best_acc = accuracy
        file = 'Pth/Stu_distillation.pth'
        torch.save(net.state_dict(), file)
    elif loss < best_loss and accuracy > best_acc:
        #损失更小且准确率更高
        best_loss = loss
        best_acc = accuracy
        file = 'Pth/Stu_distillation.pth'
        torch.save(student.state_dict(), file)
    return best_loss, best_acc

best_loss=None
best_acc=None
for i in range(1,EPOCH+1):
    train(i)
    loss,accuracy=eval_training(i)
    best_loss,best_acc=save_best(loss,accuracy,best_loss,best_acc)
print(best_acc)
