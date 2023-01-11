import torch
class EarlyStopper(object):

    def __init__(self, patience, min_delta, cumulative_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
    
        self.patience_counter = 0
        self.best_score = 0
        self.best_epoch = -1

    def is_continuable(self, epoch_i, score):
        if score <= self.best_score+self.min_delta:
            # 如果不如最好的分数。
            if not self.cumulative_delta and score > self.best_score:
                # 如果不看delta
                self.best_score = score
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return False
            return True
        else:
            self.best_score = score
            self.best_epoch = epoch_i
            self.patience_counter = 0
            return True
        
        