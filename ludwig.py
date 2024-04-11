import logging
import pprint

from load_util import load_mushroom_edibility
from ludwig.automl import autotrain
mushroom_edibility_df=load_mushroom_edibility()

auto_train_results = auto_train(
    dataset=mushroom_edibility_df,
    target='class',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)