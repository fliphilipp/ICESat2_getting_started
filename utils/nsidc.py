import os
import re
import io
import json
import math
import time
import shutil
import zipfile
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry.polygon import orient
from shapely.geometry import Polygon, mapping
from xml.etree import ElementTree as ET
from utils.utilities import get_size

#########################################################################################
def download_is2(short_name='ATL03', start_date='2018-01-01', end_date='2030-01-01', uid='userid', pwd='pwd', 
                 bbox=None, shape=None, vars_sub='all', output_dir='nsidc_outputs', email='blah@ucsd.edu', 
                 bbox_subset=None, shape_subset=None, version=None, track=None):
    """
    Download ICESat-2 data from NSIDC. 

    Parameters
    ----------
    short_name : string
        The short_name of the data product to be downloaded (e.g. ATL03, ATL06, ATL08, ATL11, ATL14, etc.)
        default: ATL03
    start_date : string
        The start date for the data query in format %Y-%m-%d, will start at 00:00:00 UTC of that date
        default: '2018-01-01' (before ICESat-2 was launched)
    end_date : string
        The end date for the data query in format %Y-%m-%d, will end at 23:59:59 UTC of that date
        default: '2030-01-01' (well into the future at the time of writing)
    uid: string
        NASA Earthdata user ID for authentication
    pwd: string
        NASA Earthdata password for authentication
    bbox: list of numeric
        Bounding box for querying for data. 
        If querying for a bounding box, supply the limits in format bbox=[longitude_min, latitude_min, longitude_max, latitude_max].
        default: None
    shape: string
        File path to a shapefile or geojson to be used for querying. 
        Needs to contain a single, valid polygon. 
        default: None
    vars_sub: list of strings
        List of variables to be used for subsetting. 
        Needs to be a valid list of paths in the HDF5 file structure of the relevant granule. 
        default: 'all' (no variable subsetting)
    output_dir: string
        File path for the outputs of the NSIDC order. 
        Specify a new output folder, or an existing output folder that has no subdirectories.
        default: 'nsidc_outputs'
    email: string
        Email address used for updating you on your order status. 
        default: 'blah@ucsd.edu'
    bbox_subset: list of numeric
        Bounding box used for subsetting the data. 
        If subsetting to a bounding box, supply the limits in format bbox=[longitude_min, latitude_min, longitude_max, latitude_max].
        default: None (in this case it defaults to the value of the input argument <bbox>, used for querying)
    shape_subset: string
        File path to a shapefile or geojson to be used for subsetting. 
        Needs to contain a single, valid polygon. 
        default: None (in this case it defaults to the value of the input argument <shape>, used for querying)

    Returns
    -------
    None
                                    
    Examples
    --------
    >>> download_is2(short_name='ATL03', start_date='2021-09-26', end_date='2021-09-28', uid='XXXXXXX', pwd='XXXXXXX', 
                     bbox=[-116.8,34.0,-116.8,34.1], shape=None, vars_sub='all', output_dir='data/ICESat2/ATL03/', email='koshla@ucsd.edu', 
                     bbox_subset=None, shape_subset=None)
    """
    
    if bbox is not None:
        bounding_box = '%.7f,%.7f,%.7f,%.7f' % tuple(bbox)
        
    if shape is not None:
        gdf = gpd.read_file(shape)

        # Simplify polygon for complex shapes in order to pass a reasonable request length to CMR. 
        # The larger the tolerance value, the more simplified the polygon.
        # Orient counter-clockwise: CMR polygon points need to be provided in counter-clockwise order. 
        # The last point should match the first point to close the polygon.
        poly = orient(gdf.simplify(0.001, preserve_topology=True).loc[0],sign=1.0)

        geojson_data = gpd.GeoSeries(poly).to_json() # Convert to geojson
        geojson_data = geojson_data.replace(' ', '') #remove spaces for API call

        #Format dictionary to polygon coordinate pairs for CMR polygon filtering
        polygon = ','.join([str(c) for xy in zip(*poly.exterior.coords.xy) for c in xy])

    start_time = '00:00:00'
    end_time = '23:59:59'
    temporal = start_date + 'T' + start_time + 'Z' + ',' + end_date + 'T' + end_time + 'Z'

    cmr_collections_url = 'https://cmr.earthdata.nasa.gov/search/collections.json'
    granule_search_url = 'https://cmr.earthdata.nasa.gov/search/granules'
    base_url = 'https://n5eil02u.ecs.nsidc.org/egi/request'

    # Get json response from CMR collection metadata
    params = {'short_name': short_name}
    response = requests.get(cmr_collections_url, params=params)
    results = json.loads(response.content)

    # Find all instances of 'version_id' in metadata and print most recent version number
    versions = [el['version_id'] for el in results['feed']['entry']]
    if version is None:
        latest_version = max(versions)
    else: 
        latest_version = '%03i' % version
    capability_url = f'https://n5eil02u.ecs.nsidc.org/egi/capabilities/{short_name}.{latest_version}.xml'

    search_params = {'short_name': short_name, 'version': latest_version, 'temporal': temporal, 'page_size': 100, 'page_num': 1}
    if bbox is not None: search_params['bounding_box'] = bounding_box
    elif shape is not None: search_params['polygon'] = polygon
    else: print('No spatial filtering criteria were given.')

    headers= {'Accept': 'application/json'}
    print("Search parameters:", search_params)

    # query for granules 
    granules = []
    headers={'Accept': 'application/json'}
    while True:
        response = requests.get(granule_search_url, params=search_params, headers=headers)
        results = json.loads(response.content)
        if len(results['feed']['entry']) == 0: break
        granules.extend(results['feed']['entry'])
        search_params['page_num'] += 1

    print('\nFound %i %s version %s granules over the search area between %s and %s.' % (len(granules), short_name, latest_version, 
                                                                              start_date, end_date))
    if len(granules) == 0: print('None')
    for result in granules[:10]:
        print('  '+result['producer_granule_id'], f', {float(result["granule_size"]):.2f} MB',sep='')
    if len(granules) > 10:
        print('  and %i more results...' % (len(granules)-10))

    # Create session to store cookie and pass credentials to capabilities url
    session = requests.session()
    s = session.get(capability_url)
    response = session.get(s.url,auth=(uid,pwd))

    root = ET.fromstring(response.content)

    #collect lists with each service option
    subagent = [subset_agent.attrib for subset_agent in root.iter('SubsetAgent')]

    # this is for getting possible variable values from the granule search
    if len(subagent) > 0 :
        # variable subsetting
        variables = [SubsetVariable.attrib for SubsetVariable in root.iter('SubsetVariable')]  
        variables_raw = [variables[i]['value'] for i in range(len(variables))]
        variables_join = [''.join(('/',v)) if v.startswith('/') == False else v for v in variables_raw] 
        variable_vals = [v.replace(':', '/') for v in variables_join]

    # make sure to only request the variables that are available
    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3
    if vars_sub == 'all':
        var_list_subsetting = ''
    else:
        var_list_subsetting = intersection(variable_vals,var_list)

    if len(subagent) < 1 :
        print('No services exist for', short_name, 'version', latest_version)
        agent = 'NO'
        coverage, Boundingshape, bbox = '', '', ''
    else:
        agent = ''
        subdict = subagent[0]
        if (subdict['spatialSubsettingShapefile'] == 'true') & (bbox is not None):
            if bbox_subset is None:
                bbox = bounding_box
            else:
                bbox = '%.7f,%.7f,%.7f,%.7f' % tuple(bbox_subset)
        else: 
            bounding_box, bbox = '', ''
        if (subdict['spatialSubsetting'] == 'true') & (shape is not None):        
            if shape_subset is None:
                Boundingshape = polygon
            else:
                gdf = gpd.read_file(shape_subset)
                poly = orient(gdf.simplify(0.05, preserve_topology=False).loc[0],sign=1.0)
                geojson_data = gpd.GeoSeries(poly).to_json()
                geojson_data = geojson_data.replace(' ', '')
                Boundingshape = ','.join([str(c) for xy in zip(*poly.exterior.coords.xy) for c in xy])
        else:
            Boundingshape, polygon = '', ''
        coverage = ','.join(var_list_subsetting)

    #Set the request mode to asynchronous if the number of granules is over 100, otherwise synchronous is enabled by default
    if len(granules) > 100:
        request_mode = 'async'
        page_size = 2000
    else: 
        page_size = 100
        request_mode = 'stream'
    #Determine number of orders needed for requests over 2000 granules. 
    page_num = math.ceil(len(granules)/page_size)

    print('  --> There will be', page_num, 'total order(s) processed for our', short_name, 'request.')
    param_dict = {'short_name': short_name, 
                  'version': latest_version, 
                  'temporal': temporal, 
                  'bbox': bbox,
                  'bounding_box': bounding_box,
                  'Boundingshape': Boundingshape, 
                  'polygon': polygon,
                  'Coverage': coverage, 
                  'page_size': page_size, 
                  'request_mode': request_mode, 
                  'agent': agent, 
                  'email': email, }

    #Remove blank key-value-pairs
    param_dict = {k: v for k, v in param_dict.items() if v != ''}

    #Convert to string
    param_string = '&'.join("{!s}={!r}".format(k,v) for (k,v) in param_dict.items())
    param_string = param_string.replace("'","")

    #Print API base URL + request parameters
    endpoint_list = [] 
    for i in range(page_num):
        page_val = i + 1
        API_request = api_request = f'{base_url}?{param_string}&page_num={page_val}'
        endpoint_list.append(API_request)

    print('\n', *endpoint_list, sep = "\n") 

    # Create an output folder if the folder does not already exist.
    path = str(os.getcwd() + '/' + output_dir)
    if not os.path.exists(path):
        os.makedirs(path)

    # Different access methods depending on request mode:
    if request_mode=='async':
        # Request data service for each page number, and unzip outputs
        for i in range(page_num):
            page_val = i + 1
            print('Order: ', page_val)

        # For all requests other than spatial file upload, use get function
            param_dict['page_num'] = page_val
            request = session.get(base_url, params=param_dict)

            print('Request HTTP response: ', request.status_code)

        # Raise bad request: Loop will stop for bad response code.
            request.raise_for_status()
            print('Order request URL: ', request.url)
            esir_root = ET.fromstring(request.content)
            print('Order request response XML content: ', request.content)

        #Look up order ID
            orderlist = []   
            for order in esir_root.findall("./order/"):
                orderlist.append(order.text)
            orderID = orderlist[0]
            print('order ID: ', orderID)

        #Create status URL
            statusURL = base_url + '/' + orderID
            print('status URL: ', statusURL)

        #Find order status
            request_response = session.get(statusURL)    
            print('HTTP response from order response URL: ', request_response.status_code)

        # Raise bad request: Loop will stop for bad response code.
            request_response.raise_for_status()
            request_root = ET.fromstring(request_response.content)
            statuslist = []
            for status in request_root.findall("./requestStatus/"):
                statuslist.append(status.text)
            status = statuslist[0]
            print('Data request ', page_val, ' is submitting...')
            print('Initial request status is ', status)

        #Continue loop while request is still processing
            while status == 'pending' or status == 'processing': 
                print('Status is not complete. Trying again.')
                time.sleep(10)
                loop_response = session.get(statusURL)

        # Raise bad request: Loop will stop for bad response code.
                loop_response.raise_for_status()
                loop_root = ET.fromstring(loop_response.content)

        #find status
                statuslist = []
                for status in loop_root.findall("./requestStatus/"):
                    statuslist.append(status.text)
                status = statuslist[0]
                print('Retry request status is: ', status)
                if status == 'pending' or status == 'processing':
                    continue

        #Order can either complete, complete_with_errors, or fail:
        # Provide complete_with_errors error message:
            if status == 'complete_with_errors' or status == 'failed':
                messagelist = []
                for message in loop_root.findall("./processInfo/"):
                    messagelist.append(message.text)
                print('error messages:')
                pprint.pprint(messagelist)

        # Download zipped order if status is complete or complete_with_errors
            if status == 'complete' or status == 'complete_with_errors':
                downloadURL = 'https://n5eil02u.ecs.nsidc.org/esir/' + orderID + '.zip'
                print('Zip download URL: ', downloadURL)
                print('Beginning download of zipped output...')
                zip_response = session.get(downloadURL)
                # Raise bad request: Loop will stop for bad response code.
                zip_response.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(zip_response.content)) as z:
                    z.extractall(path)
                print('Data request', page_val, 'is complete.')
            else: print('Request failed.')

    else:
        for i in range(page_num):
            page_val = i + 1
            print('\nOrder: ', page_val)
            print('Requesting...')
            request = session.get(base_url, params=param_dict)
            print('HTTP response from order response URL: ', request.status_code)
            request.raise_for_status()
            d = request.headers['content-disposition']
            fname = re.findall('filename=(.+)', d)
            dirname = os.path.join(path,fname[0].strip('\"'))
            print('Downloading...')
            open(dirname, 'wb').write(request.content)
            print('Data request', page_val, 'is complete.')

        # Unzip outputs
        for z in os.listdir(path): 
            if z.endswith('.zip'): 
                zip_name = path + "/" + z 
                zip_ref = zipfile.ZipFile(zip_name) 
                zip_ref.extractall(path) 
                zip_ref.close() 
                os.remove(zip_name)

    # Clean up Outputs folder by removing individual granule folders 
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            try:
                shutil.move(os.path.join(root, file), path)
            except OSError:
                pass
        for name in dirs:
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)
    
    return


##########################################################################################
def shp2geojson(shapefile, output_directory = 'geojsons/'):
    """
    Convert a shapefile to a geojson polygon file that can be used to 
    subset data from NSIDC. This already simplifies large polygons
    to reduce file size.

    Parameters
    ----------
    shapefile : string
        the path to the shapefile to convert
    output_directory : string
        the directory in which to write the geojson file

    Returns
    -------
    None

    Examples
    --------
    >>> shp2geojson_nsidc(my_shapefile.shp, output_directory = 'geojsons/')
    """    
    
    outfilename = shapefile.replace('.shp', '.geojson')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if outfilename[outfilename.rfind('/'):] != -1:
        outfilename = output_directory + outfilename[outfilename.rfind('/')+1:]
    else:
        outfilename = output_directory + outfilename

    gdf = gpd.read_file(shapefile)
    poly = orient(gdf.simplify(0.05, preserve_topology=False).loc[0],sign=1.0)
    gpd.GeoSeries(poly).to_file(outfilename, driver='GeoJSON')
    print('Wrote file: %s' % outfilename)
    return outfilename
    
##########################################################################################    
def make_granule_list(geojson, short_name='ATL03', start_date='2018-01-01', end_date='2030-01-01', list_out_name='granule-list.csv', 
                      geojson_dir_local='geojsons/', geojson_dir_remote=None):
    """
    Query for available granules over a region of interest and a start
    and end date. This will write a csv file with one column being the 
    available granule producer IDs and the other one the path to the 
    geojson file that needs to be used to subset them. 

    Parameters
    ----------
    geojson : string
        the filename of the geojson file (can also use a shapefile)
    start_date : string
        the start date in 'YYYY-MM-DD' format
    end_date : string
        the end date in 'YYYY-MM-DD' format
    list_out_name : string
        the path+filename of the csv file to be written out
    geojson_dir_local : string
        the path to the directory in which the geojson file is stored locally
    geojson_dir_remote : string
        the path to the directory in which the geojson file is stashed remotely
        if None (default) it will be the same as the local path

    Returns
    -------
    nothing, writes csv file to path given by list_out_name

    Examples
    --------
    >>> make_granule_list(my_geojson.geojson, '2021-05-01', '2021-09-15', 'auto', 
                          geojson_dir_local='geojsons/', geojson_dir_remote=None)
    """    

    start_time = '00:00:00'
    end_time = '23:59:59'
    temporal = start_date + 'T' + start_time + 'Z' + ',' + end_date + 'T' + end_time + 'Z'

    cmr_collections_url = 'https://cmr.earthdata.nasa.gov/search/collections.json'
    granule_search_url = 'https://cmr.earthdata.nasa.gov/search/granules'
    base_url = 'https://n5eil02u.ecs.nsidc.org/egi/request'

    # Get json response from CMR collection metadata
    params = {'short_name': short_name}
    response = requests.get(cmr_collections_url, params=params)
    results = json.loads(response.content)

    # Find all instances of 'version_id' in metadata and print most recent version number
    versions = [el['version_id'] for el in results['feed']['entry']]
    latest_version = max(versions)
    capability_url = f'https://n5eil02u.ecs.nsidc.org/egi/capabilities/{short_name}.{latest_version}.xml'

    # read in geojson file
    gdf = gpd.read_file(geojson_dir_local + geojson)
    poly = orient(gdf.simplify(0.05, preserve_topology=False).loc[0],sign=1.0)
    polygon = ','.join([str(c) for xy in zip(*poly.exterior.coords.xy) for c in xy])
    polygon
    search_params = {'short_name': short_name, 'version': latest_version, 'temporal': temporal, 'page_size': 100,
                     'page_num': 1,'polygon': polygon}

    # query for granules 
    granules = []
    headers={'Accept': 'application/json'}
    while True:
        response = requests.get(granule_search_url, params=search_params, headers=headers)
        results = json.loads(response.content)

        if len(results['feed']['entry']) == 0:
            break # Out of results, so break out of loop

        # Collect results and increment page_num
        granules.extend(results['feed']['entry'])
        search_params['page_num'] += 1

    print('Found %i %s version %s granules over %s between %s and %s.' % (len(granules), short_name, latest_version, 
                                                                          geojson, start_date, end_date))

    granule_list = [g['producer_granule_id'] for g in granules]
    description = [geojson.replace('.geojson','')] * len(granule_list)
    if geojson_dir_remote is None:
        geojson_remote = geojson_dir_local + geojson
    else:
        geojson_remote = geojson_dir_remote + geojson

    thisdf = pd.DataFrame({'granule': granule_list, 'geojson': geojson_remote, 'description': description})
    if list_out_name == 'auto':
        list_out_name = 'granule_lists/' + geojson.replace('.geojson', ''), + '_' + start_date[:4] + '.csv'
    thisdf.to_csv(list_out_name, header=False, index=False)
    print('Wrote file: %s' % list_out_name)
    

##########################################################################################
def download_granule(granule_id, gtxs, geojson, granule_output_path, uid, pwd, vars_sub='all'): 
    """
    Download a single ICESat-2 ATL03 granule based on its producer ID,
    subsets it to a given geojson file, and puts it into the specified
    output directory as a .h5 file. A NASA earthdata user id (uid), and
    the associated password are required. 
    (Can also provide a shapefile instead of geojson.)

    Parameters
    ----------
    granule_id : string
        the producer_granule_id for CMR search
    gtxs : string or list
        the ground tracks to request
        possible values:
            'gt1l' or 'gt1r' or 'gt2l', ... (single gtx)
            ['gt1l', 'gt3r', ...] (list of gtxs)
    geojson : string
        filepath to the geojson file used for spatial subsetting
    granule_output_path : string
        folder in which to save the subsetted granule
    uid : string
        earthdata user id
    pwd : string
        the associated password

    Returns
    -------
    nothing

    Examples
    --------
    >>> download_granule_nsidc(granule_id='ATL03_20210715182907_03381203_005_01.h5', 
                               geojson='geojsons/jakobshavn.geojson', 
                               gtxs='all'
                               granule_output_path='IS2data', 
                               uid='myuserid', 
                               pwd='mypasword')
    """
    
    short_name = 'ATL03'
    version = granule_id[30:33]
    granule_search_url = 'https://cmr.earthdata.nasa.gov/search/granules'
    capability_url = f'https://n5eil02u.ecs.nsidc.org/egi/capabilities/{short_name}.{version}.xml'
    base_url = 'https://n5eil02u.ecs.nsidc.org/egi/request'
    
    geojson_filepath = str(os.getcwd() + '/' + geojson)
    
    # set the variables for subsetting
    if vars_sub == 'default':
        vars_sub = ['/ancillary_data/atlas_sdp_gps_epoch',
                    '/orbit_info/rgt',
                    '/orbit_info/cycle_number',
                    '/orbit_info/sc_orient',
                    '/gtx/geolocation/ph_index_beg',
                    '/gtx/geolocation/segment_dist_x',
                    '/gtx/geolocation/segment_length',
                    '/gtx/geophys_corr/dem_h',
                    '/gtx/geophys_corr/geoid',
                    '/gtx/bckgrd_atlas/pce_mframe_cnt',
                    '/gtx/bckgrd_atlas/bckgrd_counts',
                    '/gtx/bckgrd_atlas/bckgrd_int_height',
                    '/gtx/bckgrd_atlas/delta_time',
                    '/gtx/heights/lat_ph',
                    '/gtx/heights/lon_ph',
                    '/gtx/heights/h_ph',
                    '/gtx/heights/dist_ph_along',
                    '/gtx/heights/delta_time',
                    '/gtx/heights/pce_mframe_cnt',
                    '/gtx/heights/ph_id_pulse',
                    '/gtx/heights/quality_ph',
                    '/ancillary_data/calibrations/dead_time/gtx']
    beam_list = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    
    if gtxs == 'all':
        var_list = sum([[v.replace('/gtx','/'+bm) for bm in beam_list] if '/gtx' in v else [v] for v in vars_sub],[])
    elif type(gtxs) == str:
        var_list = [v.replace('/gtx','/'+gtxs.lower()) if '/gtx' in v else v for v in vars_sub]
    elif type(gtxs) == list:
        var_list = sum([[v.replace('/gtx','/'+bm.lower()) for bm in gtxs] if '/gtx' in v else [v] for v in vars_sub],[])
    else: # default to requesting all beams
        var_list = sum([[v.replace('/gtx','/'+bm) for bm in beam_list] if '/gtx' in v else [v] for v in vars_sub],[])
    
    # search for the given granule
    search_params = {
        'short_name': short_name,
        'page_size': 100,
        'page_num': 1,
        'producer_granule_id': granule_id}

    granules = []
    headers={'Accept': 'application/json'}
    while True:
        response = requests.get(granule_search_url, params=search_params, headers=headers)
        results = json.loads(response.content)

        if len(results['feed']['entry']) == 0:
            # Out of results, so break out of loop
            break

        # Collect results and increment page_num
        granules.extend(results['feed']['entry'])
        search_params['page_num'] += 1

    print('\nDownloading ICESat-2 data. Found granules:')
    if len(granules) == 0:
        print('None')
        return 'none', 404
    for result in granules:
        print('  '+result['producer_granule_id'], f', {float(result["granule_size"]):.2f} MB',sep='')
        
    # Use geopandas to read in polygon file as GeoDataFrame object 
    # Note: a shapefile, KML, or almost any other vector-based spatial data format could be substituted here.
    gdf = gpd.read_file(geojson_filepath)
    
    # Simplify polygon for complex shapes in order to pass a reasonable request length to CMR. 
    # The larger the tolerance value, the more simplified the polygon.
    # Orient counter-clockwise: CMR polygon points need to be provided in counter-clockwise order. 
    # The last point should match the first point to close the polygon.
    poly = orient(gdf.simplify(0.05, preserve_topology=False).loc[0],sign=1.0)

    geojson_data = gpd.GeoSeries(poly).to_json() # Convert to geojson
    geojson_data = geojson_data.replace(' ', '') #remove spaces for API call
    
    #Format dictionary to polygon coordinate pairs for CMR polygon filtering
    polygon = ','.join([str(c) for xy in zip(*poly.exterior.coords.xy) for c in xy])
    
    print('\nInput geojson:', geojson)
    print('Simplified polygon coordinates based on geojson input:', polygon)
    
    # Create session to store cookie and pass credentials to capabilities url
    session = requests.session()
    s = session.get(capability_url)
    response = session.get(s.url,auth=(uid,pwd))

    root = ET.fromstring(response.content)

    #collect lists with each service option
    subagent = [subset_agent.attrib for subset_agent in root.iter('SubsetAgent')]
    
    # this is for getting possible variable values from the granule search
    if len(subagent) > 0 :
        # variable subsetting
        variables = [SubsetVariable.attrib for SubsetVariable in root.iter('SubsetVariable')]  
        variables_raw = [variables[i]['value'] for i in range(len(variables))]
        variables_join = [''.join(('/',v)) if v.startswith('/') == False else v for v in variables_raw] 
        variable_vals = [v.replace(':', '/') for v in variables_join]
    
    # make sure to only request the variables that are available
    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3
    if vars_sub == 'all':
        var_list_subsetting = variable_vals
    else:
        var_list_subsetting = intersection(variable_vals,var_list)
    
    if len(subagent) < 1 :
        print('No services exist for', short_name, 'version', latest_version)
        agent = 'NO'
        coverage,Boundingshape = '',''
    else:
        agent = ''
        subdict = subagent[0]
        if subdict['spatialSubsettingShapefile'] == 'true':
            Boundingshape = geojson_data
        else:
            Boundingshape = ''
        coverage = ','.join(var_list_subsetting)
        
    page_size = 100
    request_mode = 'stream'
    page_num = int(np.ceil(len(granules)/page_size))

    param_dict = {'short_name': short_name, 
                  'producer_granule_id': granule_id,
                  'version': version,  
                  'polygon': polygon,
                  'Boundingshape': Boundingshape,  
                  'Coverage': coverage, 
                  'page_size': page_size, 
                  'request_mode': request_mode, 
                  'agent': agent, 
                  'email': 'yes'}

    #Remove blank key-value-pairs
    param_dict = {k: v for k, v in param_dict.items() if v != ''}

    #Convert to string
    param_string = '&'.join("{!s}={!r}".format(k,v) for (k,v) in param_dict.items())
    param_string = param_string.replace("'","")

    #Print API base URL + request parameters
    endpoint_list = [] 
    for i in range(page_num):
        page_val = i + 1
        API_request = api_request = f'{base_url}?{param_string}&page_num={page_val}'
        endpoint_list.append(API_request)

    print('\nAPI request URL:')
    print(*endpoint_list, sep = "\n") 
    
    # Create an output folder if the folder does not already exist.
    path = str(os.getcwd() + '/' + granule_output_path)
    if not os.path.exists(path):
        os.mkdir(path)

    # Different access methods depending on request mode:
    for i in range(page_num):
        page_val = i + 1
        print('\nOrder: ', page_val)
        print('Requesting...')
        request = session.get(base_url, params=param_dict)
        print('HTTP response from order response URL: ', request.status_code)
        request.raise_for_status()
        d = request.headers['content-disposition']
        fname = re.findall('filename=(.+)', d)
        dirname = os.path.join(path,fname[0].strip('\"'))
        print('Downloading...')
        open(dirname, 'wb').write(request.content)
        print('Data request', page_val, 'is complete.')

    # Unzip outputs
    for z in os.listdir(path): 
        if z.endswith('.zip'): 
            zip_name = path + "/" + z 
            zip_ref = zipfile.ZipFile(zip_name) 
            zip_ref.extractall(path) 
            zip_ref.close() 
            os.remove(zip_name) 

    # Clean up Outputs folder by removing individual granule folders 
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            try:
                shutil.move(os.path.join(root, file), path)
            except OSError:
                pass
        for name in dirs:
            os.rmdir(os.path.join(root, name))
            
    print('\nUnzipped files and cleaned up directory.')
    print('Output data saved in:', granule_output_path)
    
    filelist = [granule_output_path+'/'+f for f in os.listdir(granule_output_path) \
                if os.path.isfile(os.path.join(granule_output_path, f)) & (granule_id in f)]
    
    if len(filelist) == 0: 
        return 'none'
    else:
        filename = filelist[0]
    print('File to process: %s (%s)' % (filename, get_size(filename)))
    
    print(filename, 'status:', request.status_code)
    return filename, request.status_code


##########################################################################################
def print_granule_stats_atl03(photon_data, bckgrd_data, ancillary, outfile=None):
    """
    Print stats from a read-in granule.
    Mostly for checking that things are working / OSG testing. 

    Parameters
    ----------
    photon_data : dict of pandas dataframes
        the first output of read_atl03()
    bckgrd_data : dict of pandas dataframes
        the second output of read_atl03()
    ancillary : dict
        the third output of read_atl03()
    outfile : string
        file path and name for the output
        if outfile=None, results are printed to stdout

    Returns
    -------
    nothing
                                    
    Examples
    --------
    >>> print_granule_stats_atl03(photon_data, bckgrd_data, ancillary, outfile='stats.txt')
    """    

    if outfile is not None: 
        import sys
        original_stdout = sys.stdout
        f = open(outfile, "w")
        sys.stdout = f

    print('\n*********************************')
    print('** GRANULE INFO AND STATISTICS **')
    print('*********************************\n')
    print(ancillary['granule_id'])
    print('RGT:', ancillary['rgt'])
    print('cycle number:', ancillary['cycle_number'])
    print('spacecraft orientation:', ancillary['sc_orient'])
    print('beam configuation:')
    for k in ancillary['gtx_beam_dict'].keys():
        print(' ', k, ': beam', ancillary['gtx_beam_dict'][k], '(%s)'%ancillary['gtx_strength_dict'][k])
    for k in photon_data.keys():
        counts = photon_data[k].count()
        nanvals = photon_data[k].isna().sum()
        maxs = photon_data[k].max()
        mins = photon_data[k].min()
        print('\nPHOTON DATA SUMMARY FOR BEAM %s'%k.upper())
        print(pd.DataFrame({'count':counts, 'nans':nanvals, 'min':mins, 'max':maxs}))
        
    if outfile is not None:
        f.close()
        sys.stdout = original_stdout
        with open(outfile, 'r') as f:
            print(f.read())
    return

################################################################################
def read_atl03(filename, geoid_h=True, gtxs_to_read='all'):
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
    dfs : dict of pandas dataframes
          photon-rate data with keys ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
          each dataframe contains the following variables
          lat : float64, latitude of the photon, degrees
          lon : float64, longitude of the photon, degrees
          h : float64, elevation of the photon (geoid correction applied if geoid_h=True), meters
          dt : float64, delta time of the photon, seconds from the ATLAS SDP GPS Epoch
          mframe : uint32, the ICESat-2 major frame that the photon belongs to
          qual : int8, quality flag 0=nominal,1=possible_afterpulse,2=possible_impulse_response_effect,3=possible_tep
          xatc : float64, along-track distance of the photon, meters
          geoid : float64, geoid correction that was applied to photon elevation (supplied if geoid_h=True), meters
    dfs_bckgrd : dict of pandas dataframes
                 photon-rate data with keys ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
                 each dataframe contains the following variables
                 pce_mframe_cnt : int64, the major frame that the data belongs to
                 bckgrd_counts : int32, number of background photons
                 bckgrd_int_height : float32, height of the background window, meters
                 delta_time : float64, Time at the start of ATLAS 50-shot sum, seconds from the ATLAS SDP GPS Epoch
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
                                    
    Examples
    --------
    >>> read_atl03(filename='processed_ATL03_20210715182907_03381203_005_01.h5', geoid_h=True)
    """
    
    print('  reading in', filename)
    granule_id = filename[filename.find('ATL03_'):(filename.find('.h5')+3)]
    
    # open file
    f = h5py.File(filename, 'r')
    
    # make dictionaries for beam data to be stored in
    dfs = {}
    dfs_bckgrd = {}
    all_beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    beams_available = [beam for beam in all_beams if "/%s/heights/" % beam in f]
    
    if gtxs_to_read=='all':
        beamlist = beams_available
    elif gtxs_to_read=='none':
        beamlist = []
    else:
        if type(gtxs_to_read)==list: beamlist = list(set(gtxs_to_read).intersection(set(beams_available)))
        elif type(gtxs_to_read)==str: beamlist = list(set([gtxs_to_read]).intersection(set(beams_available)))
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
    print('  reading in beam:', end=' ')
    for beam in beamlist:
        
        print(beam, end=' ')
        try:
            
            if gtx_strength_dict[beam]=='strong':
                ancillary['gtx_dead_time_dict'][beam] = np.mean(np.array(f['ancillary_data']['calibrations']['dead_time'][beam]['dead_time'])[:16])
            else:
                ancillary['gtx_dead_time_dict'][beam] = np.mean(np.array(f['ancillary_data']['calibrations']['dead_time'][beam]['dead_time'])[16:])
               
            #### get photon-level data
            # if "/%s/heights/" not in f: break; # 
             
            df = pd.DataFrame({'lat': np.array(f[beam]['heights']['lat_ph']),
                               'lon': np.array(f[beam]['heights']['lon_ph']),
                               'h': np.array(f[beam]['heights']['h_ph']),
                               'dt': np.array(f[beam]['heights']['delta_time']),
                               # 'conf': np.array(f[beam]['heights']['signal_conf_ph'][:,conf_landice]),
                               # not using ATL03 confidences here
                               'mframe': np.array(f[beam]['heights']['pce_mframe_cnt']),
                               'ph_id_pulse': np.array(f[beam]['heights']['ph_id_pulse']),
                               'qual': np.array(f[beam]['heights']['quality_ph'])}) 
                               # 0=nominal,1=afterpulse,2=impulse_response_effect,3=tep
            if 'weight_ph' in f[beam]['heights'].keys():
                df['weight_ph'] = np.array(f[beam]['heights']['weight_ph'])
# 
#             df_bckgrd = pd.DataFrame({'pce_mframe_cnt': np.array(f[beam]['bckgrd_atlas']['pce_mframe_cnt']),
#                                       'bckgrd_counts': np.array(f[beam]['bckgrd_atlas']['bckgrd_counts']),
#                                       'bckgrd_int_height': np.array(f[beam]['bckgrd_atlas']['bckgrd_int_height']),
#                                       'delta_time': np.array(f[beam]['bckgrd_atlas']['delta_time'])})

            #### calculate along-track distances [meters from the equator crossing] from segment-level data
            df['xatc'] = np.full_like(df.lat, fill_value=np.nan)
            ph_index_beg = np.int64(f[beam]['geolocation']['ph_index_beg']) - 1
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
                    del geoid
                else:
                    df['geoid'] = 0.0

            #### save to list of dataframes
            dfs[beam] = df
            del df 
            gc.collect()
            #Mdfs_bckgrd[beam] = df_bckgrd
        
        except:
            print('Error for {f:s} on {b:s} ... skipping:'.format(f=filename, b=beam))
            traceback.print_exc()
            
    f.close()
    print(' --> done.')
    if len(beamlist)==0:
        return beams_available, ancillary
    else:
        return beams_available, ancillary, dfs
    
def read_atl06(filename, gtxs_to_read='all'):
    # make dictionaries for beam data to be stored in
    granule_id = filename[filename.find('ATL06_'):(filename.find('.h5')+3)]
    print('  reading in', granule_id)

    # open file
    f = h5py.File(filename, 'r')
    dfs = {}

    all_beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    beams_available = [beam for beam in all_beams if "/%s/land_ice_segments/" % beam in f]

    if gtxs_to_read=='all':
        beamlist = beams_available
    elif gtxs_to_read=='none':
        beamlist = []
    else:
        if type(gtxs_to_read)==list: beamlist = list(set(gtxs_to_read).intersection(set(beams_available)))
        elif type(gtxs_to_read)==str: beamlist = list(set([gtxs_to_read]).intersection(set(beams_available)))
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
                 'date': '%s-%s-%s' % (granule_id[8:12], granule_id[12:14], granule_id[14:16]),
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
            df = pd.DataFrame({'lat': np.array(f[beam]['land_ice_segments']['latitude']),
                               'lon': np.array(f[beam]['land_ice_segments']['longitude']),
                               'h': np.array(f[beam]['land_ice_segments']['h_li']),
                               })

            #### save to list of dataframes
            dfs[beam] = df

        except:
            print('Error for {f:s} on {b:s} ... skipping:'.format(f=filename, b=beam))
            traceback.print_exc()

    f.close()
    print(' --> done.')
    if len(dfs)==0:
        return ancillary
    else:
        return ancillary, dfs