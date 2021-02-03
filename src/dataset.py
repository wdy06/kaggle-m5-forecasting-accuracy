import pandas as pd

import utils


class M5Dataset:
    def __init__(self):
        super().__init__()
        self.calendar = pd.read_csv(utils.DATA_DIR / 'calendar.csv')
        self.sell_prices = pd.read_csv(utils.DATA_DIR / 'sell_prices.csv')
        # self.main_df = pd.read_csv(
        #     utils.DATA_DIR / 'sales_train_validation.csv')
        self.main_df = pd.read_csv(
            utils.DATA_DIR / 'sales_train_evaluation.csv')
        self.submission = pd.read_csv(utils.DATA_DIR / 'sample_submission.csv')
