import torch
from torch import nn
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self, opts):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        # 在这里就是用了 ArcFace 进行计算
        self.target_id = opts.target_id
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        # torch.nn.AdaptiveAvgPool2d()
        # Applies a 2D adaptive average pooling over an input signal composed of several input planes.
        # The output is of size H x W, for any input size. 
        # The number of output features is equal to the number of input planes.
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x):
        #id_budget = 0.5
        # x is input source image, y is target image
        # y_hat is the output of x through encoder and generator
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        # size(bs, 512)
        """
        detach()
        返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，
        得到的这个Variable永远不需要计算其梯度，不具有grad。即使之后重新将它的requires_grad置为true,它也不会具有梯度grad
        这样我们就会继续使用这个新的Variable进行计算，
        后面当我们进行反向传播时，到该调用detach()的Variable就会停止，不能再继续向前进行传播
        """
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            # 余弦值越接近1，就表明夹角越接近0度，也就是两个向量越相似
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            id_logs.append({'diff_target': float(diff_target),
                            'diff_input': float(diff_input),
                            'diff_views': float(diff_views)})
            """这里在进行余弦相似度计算的时候，为什么没有除以相应向量的模长？"""
            """因为之前对目标特征已经进行了归一化处理，所以再计算余弦相似度的时候，就是两个特征向量的內积"""
            # 原本计算生成图像和输入图像对应的 target 图像的 cosine similarity
            # 现在计算生成图像和输入图像关于 Arc-Face Feature 的cosine similarity
            loss += torch.abs(self.target_id - diff_input)
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count, id_logs
