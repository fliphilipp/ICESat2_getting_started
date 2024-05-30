import os
import math
import datetime
import h5py
import numpy as np
from datetime import datetime
from datetime import timedelta
from datetime import timezone

##########################################################################################
def get_size(filename):
    size_bytes = os.path.getsize(filename)
    if size_bytes == 0: return "0 B"
    else:
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        s = round(size_bytes / math.pow(1024, i), 2)
        return "%s %s" % (s, size_name[i])

#####################################################################
def is2dt2str(lake_mean_delta_time):
    # just to make sure this is acutally just one value
    lake_mean_delta_time = np.mean(lake_mean_delta_time)
    # handle nans
    if np.isnan(lake_mean_delta_time) | (lake_mean_delta_time == np.inf):
        return np.nan
    else:
        # ATLAS SDP epoch is 2018-01-01:T00.00.00.000000 UTC, from ATL03 data dictionary
        ATLAS_SDP_epoch_datetime = datetime(2018, 1, 1, tzinfo=timezone.utc)
        ATLAS_SDP_epoch_timestamp = datetime.timestamp(ATLAS_SDP_epoch_datetime)
        lake_mean_timestamp = ATLAS_SDP_epoch_timestamp + lake_mean_delta_time
        lake_mean_datetime = datetime.fromtimestamp(lake_mean_timestamp, tz=timezone.utc)
        time_format_out = '%Y-%m-%dT%H:%M:%SZ'
        is2time = datetime.strftime(lake_mean_datetime, time_format_out)
        return is2time