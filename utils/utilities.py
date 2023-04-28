import os
import math
import datetime
import h5py

##########################################################################################
def get_size(filename):
    size_bytes = os.path.getsize(filename)
    if size_bytes == 0: return "0 B"
    else:
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        s = round(size_bytes / math.pow(1024, i), 2)
        return "%s %s" % (s, size_name[i])

##########################################################################################
def convert_time_to_string(dt):
    epoch = dt + datetime.datetime.timestamp(datetime.datetime(2018,1,1))
    return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d, %H:%M:%S")