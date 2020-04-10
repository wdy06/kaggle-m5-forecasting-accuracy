import pandas as pd

import utils


class M5Dataset:
    def __init__(self):
        super().__init__()
        self.calendar = pd.read_csv(utils.DATA_DIR / 'calendar.csv')
        # self.calendar = utils.reduce_mem_usage(self.calendar)
        self.sell_prices = pd.read_csv(utils.DATA_DIR / 'sell_prices.csv')
        # self.sell_prices = utils.reduce_mem_usage(self.sell_prices)
        self.main_df = pd.read_csv(
            utils.DATA_DIR / 'sales_train_validation.csv')
        # self.main_df = utils.reduce_mem_usage(self.main_df)
        self.submission = pd.read_csv(utils.DATA_DIR / 'sample_submission.csv')
