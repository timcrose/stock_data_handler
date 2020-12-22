import pandas as pd
import numpy as np
from copy import deepcopy
from python_utils import file_utils, time_utils, mpi_utils
import gc
from mpi4py import MPI
comm = MPI.COMM_WORLD
timestamp_offset = 1249099200

def get_timestamp_from_date_str(row):
    date_str = row[0] + ', ' + row[1].split('.')[0]
    return time_utils.get_timestamp_from_date_str(date_str, date_fmt='%m/%d/%Y, %H:%M:%S') + float('0.' + row[1].split('.')[1]) - timestamp_offset


def get_closest_idx_before_date_str(timestamp_arr, date_str, date_fmt):
    timestamp = time_utils.get_timestamp_from_date_str(date_str, date_fmt=date_fmt) - timestamp_offset
    adjusted_timestamp_arr = timestamp_arr - timestamp
    abs_adjusted_timestamp_arr = np.abs(adjusted_timestamp_arr)
    del adjusted_timestamp_arr
    gc.collect()
    idx = np.argmin(abs_adjusted_timestamp_arr)
    return idx, timestamp


def convert_csv_to_h5(sym, raw_data_dir, output_dir):
    output_dir = file_utils.os.path.join(output_dir, sym)
    file_utils.mkdir_if_DNE(output_dir)
    done_fpath = file_utils.os.path.join(output_dir, 'done')
    if file_utils.os.path.exists(done_fpath):
        return
    full_data_fpath = file_utils.os.path.join(raw_data_dir, sym + '.txt')
    
    df = pd.read_csv(full_data_fpath, names=['date', 'time', 'tick', 'bid', 'ask', 'vol'], dtype={'date':str,
                                                                                                  'time':str,
                                                                                                  'tick':np.float32,
                                                                                                  'bid':np.float32,
                                                                                                  'ask':np.float32,
                                                                                                  'vol':np.float32})
    next_date_idx_fpath  = file_utils.os.path.join(output_dir,'next_date_idx')
    if file_utils.os.path.exists(next_date_idx_fpath):
        next_date_idx = int(file_utils.read_file(next_date_idx_fpath, mode='r'))
    else:
        next_date_idx = 0
        file_utils.write_to_file(next_date_idx_fpath, '0', mode='w')
    next_date_str = df.loc[next_date_idx, 'date']
    prev_date_idx = 0
    #print('Finished reading the csv into df')
    df.loc[:,'timestamp'] = df.apply(get_timestamp_from_date_str, axis=1)
    #print('type(df.loc[0,timestamp])', type(df.loc[0,'timestamp']))
    while len(df.index) > 0 and next_date_idx <= df.index[-1]:
        #print('next_date_str', next_date_str)
        #print('next_date_idx', next_date_idx)
        #print('len before', len(df))
        #print('df before', df)
        df.drop(df.index[:next_date_idx - prev_date_idx], inplace=True)
        #print('df after', df)
        gc.collect()
        #print('Finished dropping unnecessary rows of df')
        #print('len after', len(df))
        i = next_date_idx
        while len(df.index) > 0 and i <= df.index[-1] and df.loc[i, 'date'] == next_date_str:
            i += 1

        df_today = df.head(i - next_date_idx)
        #print('df_today')
        #print(df_today)
        date_str_today = df_today.loc[next_date_idx,'date']
        timestamp_arr = df_today.loc[:, 'timestamp'].values
        idx_930_day, timestamp_930 = get_closest_idx_before_date_str(timestamp_arr, date_str_today + ', ' + '09:30:00', date_fmt='%m/%d/%Y, %H:%M:%S')
        idx_930 = idx_930_day + next_date_idx
        if timestamp_arr[idx_930_day] < timestamp_930 and idx_930 + 1 in df.index and df.loc[idx_930 + 1, 'date'] == next_date_str:
            idx_930 += 1
        idx_1600_day, timestamp_1600 = get_closest_idx_before_date_str(timestamp_arr, date_str_today + ', ' + '16:00:00', date_fmt='%m/%d/%Y, %H:%M:%S')
        idx_1600 = idx_1600_day + next_date_idx
        if timestamp_arr[idx_1600_day] > timestamp_1600 and idx_1600 - 1 in df.index and df.loc[idx_1600 - 1, 'date'] == next_date_str:
            idx_1600 -= 1
        del timestamp_arr
        gc.collect()
        day_open_price = df_today.loc[idx_930, 'tick']
        #print('type(day_open_price)', type(day_open_price))
        day_close_price = df_today.loc[idx_1600, 'tick']
        day_tick_data = df_today.loc[idx_930:idx_1600, 'tick']
        day_high_price = day_tick_data.max()
        day_low_price = day_tick_data.min()
        day_vol_data = df_today.loc[idx_930:idx_1600, 'vol']
        day_vwap_price = np.dot(day_tick_data, day_vol_data)
        del day_tick_data
        gc.collect()
        day_tot_vol = day_vol_data.sum()
        del day_vol_data
        gc.collect()
        day_vwap_price = day_vwap_price / day_tot_vol

        #print('doing premarket')
        premarket_open_price = df_today.loc[next_date_idx, 'tick']
        premarket_tick_data = df_today.loc[:idx_930 - 1, 'tick']
        premarket_high_price = premarket_tick_data.max()
        premarket_low_price = premarket_tick_data.min()
        premarket_vol_data = df_today.loc[:idx_930 - 1, 'vol']
        premarket_vwap_price = np.dot(premarket_tick_data, premarket_vol_data)
        del premarket_tick_data
        gc.collect()
        premarket_tot_vol = premarket_vol_data.sum()
        del premarket_vol_data
        gc.collect()
        if premarket_tot_vol == 0:
            premarket_vwap_price = day_open_price
        else:
            premarket_vwap_price = premarket_vwap_price / premarket_tot_vol

        afterhours_close_price = df_today.loc[i - 1, 'tick']
        afterhours_tick_data = df_today.loc[idx_1600:, 'tick']
        afterhours_high_price = afterhours_tick_data.max()
        afterhours_low_price = afterhours_tick_data.min()
        afterhours_vol_data = df_today.loc[idx_1600:, 'vol']
        afterhours_vwap_price = np.dot(afterhours_tick_data, afterhours_vol_data)
        del afterhours_tick_data
        gc.collect()
        afterhours_tot_vol = afterhours_vol_data.sum()
        del afterhours_vol_data
        gc.collect()
        if afterhours_tot_vol == 0:
            afterhours_vwap_price = day_close_price
        else:
            afterhours_vwap_price = afterhours_vwap_price / afterhours_tot_vol

        #print('doing cum vol')
        df_today.loc[:,'cum_vol'] = df_today.loc[:,'vol'].cumsum()

        #print('doing attrs')
        month, day, year = next_date_str.split('/')
        date_path_str = year + month + day
        fname = sym + '_' + date_path_str
        h5_fpath = file_utils.os.path.join(output_dir, fname + '.h5')
        attrs_dct = {'idx_930': idx_930_day,
                     'idx_1600': idx_1600_day,
                     'day_open_price': day_open_price,
                     'day_close_price': day_close_price,
                     'day_high_price': day_high_price,
                     'day_low_price': day_low_price,
                     'day_vwap_price': day_vwap_price,
                     'day_tot_vol': day_tot_vol,
                     'premarket_open_price': premarket_open_price,
                     'premarket_high_price': premarket_high_price,
                     'premarket_low_price': premarket_low_price,
                     'premarket_vwap_price': premarket_vwap_price,
                     'premarket_tot_vol': premarket_tot_vol,
                     'afterhours_close_price': afterhours_close_price,
                     'afterhours_high_price': afterhours_high_price,
                     'afterhours_low_price': afterhours_low_price,
                     'afterhours_vwap_price': afterhours_vwap_price,
                     'afterhours_tot_vol': afterhours_tot_vol,
                     'timestamp_offset': timestamp_offset}
        
        gc.collect()
        today_values = df_today.values
        del df_today
        gc.collect()
        #print('doing precision data')
        low_precision_data = today_values[:,2:-2]
        high_precision_data = today_values[:,-2:]
        #print('did precision values')
        del today_values
        gc.collect()
        low_precision_data = np.float32(low_precision_data)
        high_precision_data = np.float64(high_precision_data)
        gc.collect()
        data = np.concatenate((low_precision_data, high_precision_data), axis=1)
        del low_precision_data
        del high_precision_data
        gc.collect()
        #print('about to write data to file')
        #print('data')
        #print(data)
        #print('attrs_dct')
        #print(attrs_dct)
        # the columns of data are: tick, bid, ask, vol, timestamp, cum_vol
        file_utils.write_h5_file(h5_fpath, data, attrs_dct=attrs_dct, dset_name=None, overwrite=True, fail_if_already_exists=False, verbose=False)
        #print('wrote data to file')
        
        if len(df.index) == 0 or i == df.index[-1] + 1:
            file_utils.write_to_file(done_fpath, 'done', mode='w')
            print('All done with sym =', sym)
            return
        prev_date_idx = deepcopy(next_date_idx)
        next_date_idx = deepcopy(i)
        file_utils.write_to_file(next_date_idx_fpath, str(next_date_idx), mode='w')
        next_date_str = df.loc[i, 'date']

def main():
    raw_data_dir = '/home/trose/process_data/update_20201116/updated_tickbidask_SP500_stocks_20201116'

    output_dir = '/home/trose/process_data/update_20201116/updated_tickbidask_SP500_stocks_h5_' + str(comm.rank)
    #raw_data_dir = '/media/external_drive/Raw_Data'
    #output_dir  ='/media/external_drive/Raw_Data/AA_sample_outdir'
    csv_fpaths = file_utils.find(raw_data_dir, '*.txt', recursive=False)
    csv_fpaths = mpi_utils.split_up_list_evenly(csv_fpaths, comm.rank, comm.size)
    #####################csv_fpaths_less_than_3_GB = [csv_fpath for csv_fpath in csv_fpaths if file_utils.os.path.getsize(csv_fpath) / (1024.0 ** 3) < 3]
    #csv_fpaths_greater_than_3_GB = [csv_fpath for csv_fpath in csv_fpaths if file_utils.os.path.getsize(csv_fpath) / (1024.0 ** 3) >= 3]
    #print(' '.join(csv_fpaths_greater_than_3_GB))
    #print('csv_fpaths', csv_fpaths)
    ########################syms = [file_utils.fname_from_fpath(csv_fpath) for csv_fpath in csv_fpaths_less_than_3_GB]
    syms =  [file_utils.fname_from_fpath(csv_fpath) for csv_fpath in csv_fpaths]
    syms = [sym for sym in syms if not file_utils.os.path.exists(file_utils.os.path.join(output_dir, sym, 'done'))] #########[150:]
    #print('len(csv_fpaths), len(syms)', len(csv_fpaths), len(syms))
    #print('syms', syms)
    for sym in syms:
        convert_csv_to_h5(sym, raw_data_dir, output_dir)

main()
