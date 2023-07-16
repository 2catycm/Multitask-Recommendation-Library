import torch
class EarlyStopper(object):

    def __init__(self, patience=10, min_delta=0, cumulative_delta=False):
        """Early Stopper

        Args:
            patience (int): 多少轮没有改进就没有耐心了
            min_delta (float): 多少的改进才叫改进
            cumulative_delta (bool): 每一步的改进很小不算你改进，但是也是你未来改进评价的障碍。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta

        self.patience_counter = 0
        self.best_score = -torch.inf
        self.best_epoch = -1

    def is_continuable(self, epoch_i, score):
        if score <= self.best_score+self.min_delta:
            # 没有改进
            if not self.cumulative_delta and score > self.best_score:
                # 有微小的改进，于是更新best_score，这次改进也算你。
                self.best_score = score
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return False
            return True
        else:
            # 有所改进
            self.best_score = score
            self.best_epoch = epoch_i
            self.patience_counter = 0
            return True