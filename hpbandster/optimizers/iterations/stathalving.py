from hpbandster.core.base_iteration import BaseIteration
import numpy as np

class model:
    def __init__(self, record, window_length=2, fixed=False):
        self.mean = 0
        self.std = 1
        self.distribution = 'normal'
        self.window_length = window_length
        self.record = record
        self.sliding_record = []
        self.std_numer = 1
        self.std_denom = 1
        self.t = 0
        self.time_decay = False
        self.fixed = fixed
        for value in record:
            self.getReward(value)

    def prediction(self, predict_length=1, alpha=1, time_decay=False):
        '''
        Create a max distribution based on model and sample from it.
        '''
        # print(self.mean + self.std * (np.sqrt((self.window_length - 1))))
        # print(predict_length)
        value = self.mean + self.std * np.sqrt(2 * np.log(predict_length))
        try:
            if len(value) > 0:
                return value[0]
        except:
            return value

    def getReward(self, value):
        """
        update mean/variance according to input value.
        """
        # self.mean = max(self.record[-self.window_length:])
        # self.mean = np.mean(self.record[-10:])
        if len(self.record) > self.window_length:
            if not self.fixed:
                std = np.std(self.record[-self.window_length:])
                self.std = 0.05 * std + 0.95 * self.std
                self.mean = 0.05 * value + 0.95 * self.mean
            else:
                self.std = np.std(self.record)
                self.mean = np.mean(self.record)
        else:
            self.std = 1


class stathalving(BaseIteration):
    def __init__(self, *args, **kwargs):
           self.stat = True
           super(stathalving, self).__init__(*args, **kwargs)
           
           # print(args[0])

    def _advance_to_next_stage(self, config_ids, acc_records, predict_budgets):
        """
        StatHalving computes prediction based on the current loss record.
        """
        models = [model(np.array(acc_record)) for acc_record in acc_records]
        prediction = [model.prediction(predict_length=predict_budgets) for model in models]
        ranks = np.argsort(np.argsort(prediction))
        return (ranks > (len(ranks) - self.num_configs[self.stage] - 1))
    