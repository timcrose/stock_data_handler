import sys
sys.path.append('D:\python_scripts')
from python_utils import file_utils, time_utils
import datetime
import numpy as np


def main():
    h5_input_dir = r'F:\Raw_Data\kibot\test'
    resample_interval = 60
    # the columns of data are: tick, bid, ask, vol, timestamp, cum_vol
    h5_fpaths = file_utils.find(h5_input_dir, '*.h5')[1:]
    for h5_fpath in h5_fpaths:
        fname = file_utils.fname_from_fpath(h5_fpath)
        todays_date_str = fname[fname.find('_') + 1:]
        
        date_str_0700 = todays_date_str + ', 07:00:00'
        timestamp_0700 = time_utils.get_timestamp_from_date_str(date_str_0700, date_fmt='%Y%m%d, %I:%M:%S')
        timestamp_2000 = timestamp_0700 + 46800
        dt_2000 = datetime.datetime.fromtimestamp(timestamp_2000)
        data, attrs_dct = file_utils.read_h5_file(h5_fpath)
        print(data[:10])
        resampled_data = []
        timestamp_arr = data[:,4] + attrs_dct['timestamp_offset']
        dt_i = datetime.datetime.fromtimestamp(timestamp_0700)
        open_price, high_price, low_price, close_price, volume = 0,0,0,0,0
        open_bool = True
        i = 0
        dt = datetime.datetime.fromtimestamp(timestamp_arr[i])
        for i in range(10):
            dt = datetime.datetime.fromtimestamp(timestamp_arr[i])    
            print(dt)
        return
        while dt_i < dt_2000:
            while dt <= dt_i - datetime.timedelta(seconds=resample_interval):
                i += 1
                dt = datetime.datetime.fromtimestamp(timestamp_arr[i])
            if dt > dt_i:
                # No data exists in the time interval given by dt_i. V (volume)
                # is 0.
                # Create the resampled OHLCV using the data in the previous 
                # time interval (if it exists). If no previous time interval
                # exists, use the first data point as the OHLC value.
                if i == 0:
                    open_price = data[i, 0]
                    high_price = data[i, 0]
                    low_price = data[i, 0]
                    close_price = data[i, 0]
                    volume = 0
                elif open_bool:
                    open_price = data[i - 1, 0]
                    high_price = data[i - 1, 0]
                    low_price = data[i - 1, 0]
                    close_price = data[i - 1, 0]
                    volume = 0
                resampled_data.append([open_price, high_price, low_price, 
                        close_price, volume])
                
                open_price, high_price, low_price, close_price, volume = 0,0,0,0,0
                open_bool = True
                dt_i += datetime.timedelta(seconds=resample_interval)
                continue
            
            if open_bool:
                open_price = data[i, 0]
                open_bool = False
            if data[i, 0] > high_price:
                high_price = data[i, 0]
            if data[i, 0] < low_price:
                low_price = data[i, 0]
            volume += data[i, 3]
            i += 1
            if i >= len(timestamp_arr):
                resampled_data.append([open_price, high_price, low_price, 
                        close_price, volume])
                
                dt_i += datetime.timedelta(seconds=resample_interval)
                while dt_i < dt_2000:
                    open_price = data[i - 1, 0]
                    high_price = data[i - 1, 0]
                    low_price = data[i - 1, 0]
                    close_price = data[i - 1, 0]
                    volume = 0
                    
                    resampled_data.append([open_price, high_price, low_price, 
                            close_price, volume])
                    
                    dt_i += datetime.timedelta(seconds=resample_interval)
            else:
                dt = datetime.datetime.fromtimestamp(timestamp_arr[i])
        attrs_dct['resample_interval'] = resample_interval
        #file_utils.write_h5_file('test.h5', 
        #        np.array(resampled_data), attrs_dct=attrs_dct)
        
        for datum in resampled_data[:9]:
            print(datum)
                
if __name__ == '__main__':
    main()