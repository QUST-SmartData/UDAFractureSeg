import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from skimage import measure


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

    
def isolated_region_size(img):

    img_binary = torch.argmax(img, dim=1).byte()

    # Use skimage.measure.label to label connected components
    labeled_pred, num_labels = measure.label(img_binary.cpu().numpy(), background=0, return_num=True)

    # Count the size of isolated regions
    isolated_size = 0
    for region in measure.regionprops(labeled_pred):
        if region.area == 1:
            isolated_size += 1

    return isolated_size


def isolation_loss(pred, target):
    # Convert to binary image
    pred = F.softmax(pred, dim=1)

    pred_isolated_size = isolated_region_size(pred)
    target_isolated_size = isolated_region_size(target)

    return abs(pred_isolated_size - target_isolated_size)
