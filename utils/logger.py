class Logger:
    def __init__(self, log_dict: dict = {}):
        self.logs = log_dict
        self.count = 0

    def update(self, log_dict: dict):
        for k, v in log_dict.items():
            self.logs[k] = (self.logs.get(k, 0) * self.count + v) / (self.count + 1)
        self.count += 1
