from model import *

class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = FeatureExtractVGG().cuda().eval()
        self.tv_loss =TVLoss()
        self.mse_loss = nn.MSELoss()
        # self.feature_extractor.eval()

    def forward(self,SR_img,HR_img):
        SR_feature = self.feature_extractor(SR_img)
        HR_feature = self.feature_extractor(HR_img)

        loss = F.mse_loss(SR_feature,HR_feature)
        loss_tv =self.tv_loss(SR_img)
        loss_image = self.mse_loss(SR_img,HR_img)
        return 0.006* loss 
    
class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, discriminator_output):
        return -torch.log(torch.mean(discriminator_output)+1e-9)

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]