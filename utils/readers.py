import os
import re
import json
import h5py
import math
import shutil
import zipfile
import shapely
import requests
import datetime
import traceback
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry.polygon import orient
from shapely.geometry import Polygon, mapping
from xml.etree import ElementTree as ET

##########################################################################################
def convert_time_to_string(dt):
    epoch = dt + datetime.datetime.timestamp(datetime.datetime(2018,1,1))
    return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d, %H:%M:%S")

##########################################################################################
def read_atl03(filename, geoid_h=True, gtxs_to_read='all', verbose=False):
    """
    Read in an ATL03 granule. 

    Parameters
    ----------
    filename : string
        the file path of the granule to be read in
    geoid_h : boolean
        whether to include the ATL03-supplied geoid correction for photon heights

    Returns
    -------
    ancillary : dictionary with the following keys:
                granule_id : string, the producer granule id, extracted from filename
                atlas_sdp_gps_epoch : float64, reference GPS time for ATLAS in seconds [1198800018.0]
                rgt : int16, the reference ground track number
                cycle_number : int8, the ICESat-2 cycle number of the granule
                sc_orient : the spacecraft orientation (usually 'forward' or 'backward')
                gtx_beam_dict : dictionary of the ground track / beam number configuration 
                                example: {'gt1l': 6, 'gt1r': 5, 'gt2l': 4, 'gt2r': 3, 'gt3l': 2, 'gt3r': 1}
                gtx_strength_dict': dictionary of the ground track / beam strength configuration
                                    example: {'gt1l': 'weak','gt1r': 'strong','gt2l': 'weak', ... }
    dfs : dict of pandas dataframes
          photon-rate data with keys ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r'] (if all beams are availaible)
          each dataframe contains the following variables
          lat : float64, latitude of the photon, degrees
          lon : float64, longitude of the photon, degrees
          h : float64, elevation of the photon (geoid correction applied if geoid_h=True), meters
          dt : float64, delta time of the photon, seconds from the ATLAS SDP GPS Epoch
          mframe : uint32, the ICESat-2 major frame that the photon belongs to
          qual : int8, quality flag 0=nominal,1=possible_afterpulse,2=possible_impulse_response_effect,3=possible_tep
          xatc : float64, along-track distance of the photon, meters
          geoid : float64, geoid correction that was applied to photon elevation (supplied if geoid_h=True), meters

                                    
    Examples
    --------
    >>> read_atl03(filename='processed_ATL03_20210715182907_03381203_005_01.h5', geoid_h=True)
    """
    
    if verbose: print('  reading in', filename)
    granule_id = filename[filename.find('ATL03_'):(filename.find('.h5')+3)]
    
    # open file
    f = h5py.File(filename, 'r')
    
    # make dictionaries for beam data to be stored in
    dfs = {}
    beams_available = [x for x in list(f.keys()) if 'gt' in x]
    if gtxs_to_read=='all':
        beamlist = beams_available
    elif gtxs_to_read=='none':
        beamlist = []
    else:
        if type(gtxs_to_read)==list: beamlist = gtxs_to_read
        elif type(gtxs_to_read)==str: beamlist = [gtxs_to_read]
        else: beamlist = beams_available
    
    conf_landice = 3 # index for the land ice confidence
    
    orient = f['orbit_info']['sc_orient'][0]
    def orient_string(sc_orient):
        if sc_orient == 0:
            return 'backward'
        elif sc_orient == 1:
            return 'forward'
        elif sc_orient == 2:
            return 'transition'
        else:
            return 'error'
        
    orient_str = orient_string(orient)
    gtl = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    beam_strength_dict = {k:['weak','strong'][k%2] for k in np.arange(1,7,1)}
    if orient_str == 'forward':
        bl = np.arange(6,0,-1)
        gtx_beam_dict = {k:v for (k,v) in zip(gtl,bl)}
        gtx_strength_dict = {k:beam_strength_dict[gtx_beam_dict[k]] for k in gtl}
    elif orient_str == 'backward':
        bl = np.arange(1,7,1)
        gtx_beam_dict = {k:v for (k,v) in zip(gtl,bl)}
        gtx_strength_dict = {k:beam_strength_dict[gtx_beam_dict[k]] for k in gtl}
    else:
        gtx_beam_dict = {k:'undefined' for k in gtl}
        gtx_strength_dict = {k:'undefined' for k in gtl}
        

    ancillary = {'granule_id': granule_id,
                 'atlas_sdp_gps_epoch': f['ancillary_data']['atlas_sdp_gps_epoch'][0],
                 'rgt': f['orbit_info']['rgt'][0],
                 'cycle_number': f['orbit_info']['cycle_number'][0],
                 'sc_orient': orient_str,
                 'gtx_beam_dict': gtx_beam_dict,
                 'gtx_strength_dict': gtx_strength_dict,
                 'gtx_dead_time_dict': {}}

    # loop through all beams
    if verbose: print('  reading in beam:', end=' ')
    for beam in beamlist:
        
        if verbose: print(beam, end=' ')
        try:
            try:
                if gtx_strength_dict[beam]=='strong':
                    ancillary['gtx_dead_time_dict'][beam] = np.mean(np.array(f['ancillary_data']['calibrations']['dead_time'][beam]['dead_time'])[:16])
                else:
                    ancillary['gtx_dead_time_dict'][beam] = np.mean(np.array(f['ancillary_data']['calibrations']['dead_time'][beam]['dead_time'])[16:])
            except:
                pass
             
            df = pd.DataFrame({'lat': np.array(f[beam]['heights']['lat_ph']),
                               'lon': np.array(f[beam]['heights']['lon_ph']),
                               'h': np.array(f[beam]['heights']['h_ph']),
                               'dt': np.array(f[beam]['heights']['delta_time']),
                               'dist_ph_across': np.array(f[beam]['heights']['dist_ph_across']),
                               'conf': np.array(f[beam]['heights']['signal_conf_ph'][:,conf_landice]),
                               # not using ATL03 confidences here
                               'mframe': np.array(f[beam]['heights']['pce_mframe_cnt']),
                               'ph_id_pulse': np.array(f[beam]['heights']['ph_id_pulse']),
                               'qual': np.array(f[beam]['heights']['quality_ph'])}) 
                               # 0=nominal,1=afterpulse,2=impulse_response_effect,3=tep

            #### calculate along-track distances [meters from the equator crossing] from segment-level data
            df['xatc'] = np.full_like(df.lat, fill_value=np.nan)
            ph_index_beg = np.int32(f[beam]['geolocation']['ph_index_beg']) - 1
            segment_dist_x = np.array(f[beam]['geolocation']['segment_dist_x'])
            segment_length = np.array(f[beam]['geolocation']['segment_length'])
            valid = ph_index_beg>=0 # need to delete values where there's no photons in the segment (-1 value)
            df.loc[ph_index_beg[valid], 'xatc'] = segment_dist_x[valid]
            df.xatc.fillna(method='ffill',inplace=True)
            df.xatc += np.array(f[beam]['heights']['dist_ph_along'])

            #### now we can filter out TEP (we don't do IRF / afterpulses because it seems to not be very good...)
            df.query('qual < 3',inplace=True) 
            # df.drop(columns=['qual'], inplace=True)

            #### sort by along-track distance (for interpolation to work smoothly)
            df.sort_values(by='xatc',inplace=True)
            df.reset_index(inplace=True, drop=True)

            if geoid_h:
                #### interpolate geoid to photon level using along-track distance, and add to elevation
                geophys_geoid = np.array(f[beam]['geophys_corr']['geoid'])
                geophys_geoid_x = segment_dist_x+0.5*segment_length
                valid_geoid = geophys_geoid<1e10 # filter out INVALID_R4B fill values
                geophys_geoid = geophys_geoid[valid_geoid]
                geophys_geoid_x = geophys_geoid_x[valid_geoid]
                # hacky fix for no weird stuff happening if geoid is undefined everywhere
                if len(geophys_geoid>5):
                    geoid = np.interp(np.array(df.xatc), geophys_geoid_x, geophys_geoid)
                    df['h'] = df.h - geoid
                    df['geoid'] = geoid
                else:
                    df['geoid'] = 0.0
            
            # re-sort by delta_time
            df.sort_values(by='dt',inplace=True)
            df.reset_index(inplace=True, drop=True)

            #### save to list of dataframes
            dfs[beam] = df
        
        except:
            print('Error for {f:s} on {b:s} ... skipping:'.format(f=filename, b=beam))
            traceback.print_exc()
            
    f.close()
    if verbose: print(' --> done.')
    return ancillary, dfs

##########################################################################################
def read_atl08(filename, gtxs_to_read='all'):
    # make dictionaries for beam data to be stored in
    granule_id = filename[filename.find('ATL08'):(filename.find('.h5')+3)]
    print('  reading in', granule_id)

    # open file
    f = h5py.File(filename, 'r')
    dfs_20 = {}
    dfs_100 = {}

    beams_available = [x for x in list(f.keys()) if 'gt' in x]
    if gtxs_to_read=='all':
        beamlist = beams_available
    elif gtxs_to_read=='none':
        beamlist = []
    else:
        if type(gtxs_to_read)==list: beamlist = gtxs_to_read
        elif type(gtxs_to_read)==str: beamlist = [gtxs_to_read]
        else: beamlist = beams_available

    orient = f['orbit_info']['sc_orient'][0]
    def orient_string(sc_orient):
        if sc_orient == 0:
            return 'backward'
        elif sc_orient == 1:
            return 'forward'
        elif sc_orient == 2:
            return 'transition'
        else:
            return 'error'

    orient_str = orient_string(orient)
    gtl = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    beam_strength_dict = {k:['weak','strong'][k%2] for k in np.arange(1,7,1)}
    if orient_str == 'forward':
        bl = np.arange(6,0,-1)
        gtx_beam_dict = {k:v for (k,v) in zip(gtl,bl)}
        gtx_strength_dict = {k:beam_strength_dict[gtx_beam_dict[k]] for k in gtl}
    elif orient_str == 'backward':
        bl = np.arange(1,7,1)
        gtx_beam_dict = {k:v for (k,v) in zip(gtl,bl)}
        gtx_strength_dict = {k:beam_strength_dict[gtx_beam_dict[k]] for k in gtl}
    else:
        gtx_beam_dict = {k:'undefined' for k in gtl}
        gtx_strength_dict = {k:'undefined' for k in gtl}

    ancillary = {'granule_id': granule_id,
                 'date': '%s-%s-%s' % (granule_id[6:10], granule_id[10:12], granule_id[12:14]),
                 'atlas_sdp_gps_epoch': f['ancillary_data']['atlas_sdp_gps_epoch'][0],
                 'rgt': f['orbit_info']['rgt'][0],
                 'cycle_number': f['orbit_info']['cycle_number'][0],
                 'sc_orient': orient_str,
                 'gtx_beam_dict': gtx_beam_dict,
                 'gtx_strength_dict': gtx_strength_dict
                }

    # loop through all beams
    print('  reading in beam:', end=' ')
    for beam in beamlist:

        print(beam, end=' ')
        try:
            n_data_points = len(np.array(f[beam]['land_segments']['latitude_20m']).flatten())
            df_20 = pd.DataFrame({'lat': np.array(f[beam]['land_segments']['latitude_20m']).flatten(),
                                  'lon': np.array(f[beam]['land_segments']['longitude_20m']).flatten(),
                                  'xatc': np.linspace(0, (n_data_points-1)*20.0, n_data_points),
                                  'h': np.array(f[beam]['land_segments']['terrain']['h_te_best_fit_20m']).flatten()
                                 })
            
            df_100 = pd.DataFrame({'delta_time': np.array(f[beam]['land_segments']['delta_time']),
                                   'lat': np.array(f[beam]['land_segments']['latitude']),
                                   'lon': np.array(f[beam]['land_segments']['longitude']),
                                   'xatc': np.linspace(0, (n_data_points-1)*100.0, n_data_points),
                                   'h': np.array(f[beam]['land_segments']['terrain']['h_te_best_fit'])
                                  })
            #### save to list of dataframes
            dfs_20[beam] = df_20
            dfs_100[beam] = df_100

        except:
            print('Error for {f:s} on {b:s} ... skipping:'.format(f=filename, b=beam))
            traceback.print_exc()

    f.close()
    print(' --> done.')
    
    return ancillary, dfs_100, dfs_20

##########################################################################################
def read_atl06(filename, gtxs_to_read='all', verbose=False):
    # make dictionaries for beam data to be stored in
    granule_id = filename[filename.find('ATL06_'):(filename.find('.h5')+3)]
    if verbose: print('  reading in', granule_id)

    # open file
    f = h5py.File(filename, 'r')
    dfs = {}

    beams_available = [x for x in list(f.keys()) if 'gt' in x]
    if gtxs_to_read=='all':
        beamlist = beams_available
    elif gtxs_to_read=='none':
        beamlist = []
    else:
        if type(gtxs_to_read)==list: beamlist = gtxs_to_read
        elif type(gtxs_to_read)==str: beamlist = [gtxs_to_read]
        else: beamlist = beams_available

    orient = f['orbit_info']['sc_orient'][0]
    def orient_string(sc_orient):
        if sc_orient == 0:
            return 'backward'
        elif sc_orient == 1:
            return 'forward'
        elif sc_orient == 2:
            return 'transition'
        else:
            return 'error'

    orient_str = orient_string(orient)
    gtl = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    beam_strength_dict = {k:['weak','strong'][k%2] for k in np.arange(1,7,1)}
    if orient_str == 'forward':
        bl = np.arange(6,0,-1)
        gtx_beam_dict = {k:v for (k,v) in zip(gtl,bl)}
        gtx_strength_dict = {k:beam_strength_dict[gtx_beam_dict[k]] for k in gtl}
    elif orient_str == 'backward':
        bl = np.arange(1,7,1)
        gtx_beam_dict = {k:v for (k,v) in zip(gtl,bl)}
        gtx_strength_dict = {k:beam_strength_dict[gtx_beam_dict[k]] for k in gtl}
    else:
        gtx_beam_dict = {k:'undefined' for k in gtl}
        gtx_strength_dict = {k:'undefined' for k in gtl}

    ancillary = {'granule_id': granule_id,
                 'date': '%s-%s-%s' % (granule_id[6:10], granule_id[10:12], granule_id[12:14]),
                 'atlas_sdp_gps_epoch': f['ancillary_data']['atlas_sdp_gps_epoch'][0],
                 'rgt': f['orbit_info']['rgt'][0],
                 'cycle_number': f['orbit_info']['cycle_number'][0],
                 'sc_orient': orient_str,
                 'gtx_beam_dict': gtx_beam_dict,
                 'gtx_strength_dict': gtx_strength_dict
                }

    # loop through all beams
    if verbose: print('  reading in beam:', end=' ')
    for beam in beamlist:

        if verbose: print(beam, end=' ')
        try:
            if 'land_ice_segments' in f[beam].keys():
                df = pd.DataFrame({'delta_time': np.array(f[beam]['land_ice_segments']['delta_time']),
                                   'lat': np.array(f[beam]['land_ice_segments']['latitude']),
                                   'lon': np.array(f[beam]['land_ice_segments']['longitude']),
                                   'xatc': np.array(f[beam]['land_ice_segments']['ground_track']['x_atc']),
                                   'yatc': np.array(f[beam]['land_ice_segments']['ground_track']['y_atc']),
                                   'dh_fit_dy': np.array(f[beam]['land_ice_segments']['fit_statistics']['dh_fit_dy']),
                                   'h': np.array(f[beam]['land_ice_segments']['h_li']),
                                   'geoid_h': np.array(f[beam]['land_ice_segments']['dem']['geoid_h']),
                                   'qual_summary': np.array(f[beam]['land_ice_segments']['atl06_quality_summary']),
                                  })
                #### save to list of dataframes
                dfs[beam] = df

        except:
            print('Error for {f:s} on {b:s} ... skipping:'.format(f=filename, b=beam))
            if verbose: traceback.print_exc()

    f.close()
    if verbose: print(' --> done.')
    
    return ancillary, dfs