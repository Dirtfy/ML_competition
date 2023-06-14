class EarlyStop():
    def __init__(self, min_epoch=5):

        self.min_epoch = min_epoch

        self.test_acc_list = []
        self.train_acc_list = []
        self.test_loss_list = []
        self.train_loss_list = []

    def stop(self):
        if len(self.test_acc_list) > self.min_epoch:
            return self.test_acc_list[-1] <= (sum(self.test_acc_list[-(self.min_epoch+1):-1]) / self.min_epoch)