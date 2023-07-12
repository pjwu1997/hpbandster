import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis


models = ['HYPERBAND', 'STAT_BAND', 'STAT_BOHB', 'BOHB']
for model in models:
        
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(f'{model}/1')

    # get all executed runs
    all_runs = result.get_all_runs()

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()


    # Here is how you get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()

    # let's grab the run on the highest budget 
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]


    # We have access to all information: the config, the loss observed during
    #optimization, and all the additional information
    inc_loss = inc_run.loss
    inc_config = id2conf[inc_id]['config']
    inc_test_loss = inc_run.info['test accuracy']
    print(model)
    print('Best found configuration:')
    print(inc_config)
    print('It achieved accuracies of %f (validation) and %f (test).'%(1-inc_loss, inc_test_loss))