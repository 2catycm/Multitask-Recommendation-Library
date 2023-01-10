import torch
class EarlyStopper(object):

    def __init__(self, patience, min_delta, cumulative_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
    
        self.patience_counter = 0
        self.best_score = 0

    def is_continuable(self, model, score):
        if score <= self.best_score+self.min_delta:
            self.best_accuracy = score
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return False
        else:
            self.best_score = score
            self.patience_counter = 0
            return True
        
        