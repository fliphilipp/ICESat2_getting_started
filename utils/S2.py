import os
os.environ['USE_PYGEOS'] = '0'
import re
import ee
import geemap
import json
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib 
import matplotlib.pylab as plt
from matplotlib import cm
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import rasterio as rio
from rasterio import plot as rioplot
from rasterio import warp
import traceback
from shapely.geometry import Point, Polygon, LineString
from sliderule import sliderule, earthdata, h5, raster, icesat2, gedi
from pyproj import CRS

from utils.nsidc import download_is2, read_atl03, read_atl06
from ed.edcreds import getedcreds

try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()


#####################################################################
def get_bbox(lon, lat, buffer):
    local_crs_stere = CRS("+proj=tmerc +lat_0={0} +lon_0={1} +datum=WGS84 +units=m".format(lat, lon))
    roi = gpd.GeoSeries(Point(lon, lat),crs='EPSG:4326').to_crs(local_crs_stere).buffer(buffer).to_crs('EPSG:4326')
    coords = roi.loc[0].exterior.coords.xy
    bbox = [np.min(coords[0]), np.min(coords[1]), np.max(coords[0]), np.max(coords[1])]
    return bbox


#####################################################################
def add_graticule(img, ax_img):
    from utils.curve_intersect import intersection
    latlon_bbox = warp.transform(img.crs, {'init': 'epsg:4326'}, 
                                 [img.bounds[i] for i in [0,2,2,0,0]], 
                                 [img.bounds[i] for i in [1,1,3,3,1]])
    min_lat = np.min(latlon_bbox[1])
    max_lat = np.max(latlon_bbox[1])
    min_lon = np.min(latlon_bbox[0])
    max_lon = np.max(latlon_bbox[0])
    latdiff = max_lat-min_lat
    londiff = max_lon-min_lon
    diffs = np.array([0.0001, 0.0002, 0.00025, 0.0004, 0.0005,
                      0.001, 0.002, 0.0025, 0.004, 0.005, 
                      0.01, 0.02, 0.025, 0.04, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 1, 2])
    latstep = np.min(diffs[diffs>latdiff/8])
    lonstep = np.min(diffs[diffs>londiff/8])
    minlat = np.floor(min_lat/latstep)*latstep
    maxlat = np.ceil(max_lat/latstep)*latstep
    minlon = np.floor(min_lon/lonstep)*lonstep
    maxlon = np.ceil(max_lon/lonstep)*lonstep

    # plot meridians and parallels
    xl = (img.bounds.left, img.bounds.right)
    yl = (img.bounds.bottom, img.bounds.top)
    meridians = np.arange(minlon,maxlon, step=lonstep)
    parallels = np.arange(minlat,maxlat, step=latstep)
    latseq = np.linspace(minlat,maxlat,200)
    lonseq = np.linspace(minlon,maxlon,200)
    gridcol = 'k'
    gridls = ':'
    gridlw = 0.5
    topline = [[xl[0],xl[1]],[yl[1],yl[1]]]
    bottomline = [[xl[0],xl[1]],[yl[0],yl[0]]]
    leftline = [[xl[0],xl[0]],[yl[0],yl[1]]]
    rightline = [[xl[1],xl[1]],[yl[0],yl[1]]]
    for me in meridians:
        gr_trans = warp.transform({'init': 'epsg:4326'},img.crs,me*np.ones_like(latseq),latseq)
        deglab = ' %.10g°E' % me if me >= 0 else ' %.10g°W' % -me
        intx,inty = intersection(topline[0], topline[1], gr_trans[0], gr_trans[1])
        if len(intx) > 0:
            intx = intx[0]
            inty = inty[0]
            ax_img.text(intx, inty, deglab, fontsize=6, color='gray',verticalalignment='bottom',horizontalalignment='center',
                    rotation='vertical')
        thislw = gridlw
        ax_img.plot(gr_trans[0],gr_trans[1],c=gridcol,ls=gridls,lw=thislw,alpha=0.5)
    for pa in parallels:
        gr_trans = warp.transform({'init': 'epsg:4326'},img.crs,lonseq,pa*np.ones_like(lonseq))
        thislw = gridlw
        deglab = ' %.10g°N' % pa if pa >= 0 else ' %.10g°S' % -pa
        intx,inty = intersection(rightline[0], rightline[1], gr_trans[0], gr_trans[1])
        if len(intx) > 0:
            intx = intx[0]
            inty = inty[0]
            ax_img.text(intx, inty, deglab, fontsize=6, color='gray',verticalalignment='center',horizontalalignment='left')
        ax_img.plot(gr_trans[0],gr_trans[1],c=gridcol,ls=gridls,lw=thislw,alpha=0.5)
        ax_img.set_xlim(xl)
        ax_img.set_ylim(yl)


#####################################################################
def get_sentinel2_cloud_collection(lon, lat, date_time, days_buffer, buffer_m=2500, CLD_PRB_THRESH=40, BUFFER=100):
    # create the area of interest for cloud likelihood assessment
    point_of_interest = ee.Geometry.Point(lon, lat)
    area_of_interest = point_of_interest.buffer(buffer_m)

    datetime_requested = datetime.strptime(date_time, '%Y-%m-%dT%H:%M:%SZ')
    start_date = (datetime_requested - timedelta(days=days_buffer)).strftime('%Y-%m-%dT%H:%M:%S')
    end_date = (datetime_requested + timedelta(days=days_buffer)).strftime('%Y-%m-%dT%H:%M:%S')
    print('Looking for Sentinel-2 images from %s to %s' % (start_date, end_date), end=' ')

    # Import and filter S2 SR HARMONIZED
    s2_sr_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(area_of_interest)
        .filterDate(start_date, end_date))

    # Import and filter s2cloudless.
    s2_cloudless_collection = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(area_of_interest)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    cloud_collection = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_collection,
        'secondary': s2_cloudless_collection,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

    cloud_collection = cloud_collection.map(lambda img: img.addBands(ee.Image(img.get('s2cloudless')).select('probability')))

    def set_is2_cloudiness(img, aoi=area_of_interest):
        cloudprob = img.select(['probability']).reduceRegion(reducer=ee.Reducer.mean(), 
                                                             geometry=aoi, 
                                                             bestEffort=True, 
                                                             maxPixels=1e6)
        return img.set('ground_track_cloud_prob', cloudprob.get('probability'))

    cloud_collection = cloud_collection.map(set_is2_cloudiness)

    return cloud_collection


#####################################################################
def plotS2cloudfree(lon, lat, date_time, buffer_m=2500, max_cloud_prob=15, gamma_value=1.8, 
                    imagery_filename='imagery/my-satellite-image.tif', plot_filename='plots/sentinel2_cloudfree.jpg', 
                    ax=None, download_imagery=True):
    
    datetime_is2 = datetime.strptime(date_time, '%Y-%m-%dT%H:%M:%SZ')
    
    if download_imagery or (not os.path.isfile(imagery_filename)):
        days_buffer = 10
        collection_size = 0
        if days_buffer > 200:
            days_buffer = 200
        increment_days = days_buffer
        while (collection_size<10) & (days_buffer <= 200):

            collection = get_sentinel2_cloud_collection(lon, lat, date_time, days_buffer=days_buffer)

            # filter collection to only images that are (mostly) cloud-free along the ICESat-2 ground track
            cloudfree_collection = collection.filter(ee.Filter.lt('ground_track_cloud_prob', max_cloud_prob))

            collection_size = cloudfree_collection.size().getInfo()
            if collection_size == 1: 
                print('--> there is %i cloud-free image.' % collection_size)
            elif collection_size > 1: 
                print('--> there are %i cloud-free images.' % collection_size)
            else:
                print('--> there are not enough cloud-free images: widening date range...')
            days_buffer += increment_days

        # get the time difference between ICESat-2 and Sentinel-2 and sort by it 
        is2time = date_time
        def set_time_difference(img, is2time=is2time):
            timediff = ee.Date(is2time).difference(img.get('system:time_start'), 'second').abs()
            return img.set('timediff', timediff)
        cloudfree_collection = cloudfree_collection.map(set_time_difference).sort('timediff')

        # create a region around the ground track over which to download data
        point_of_interest = ee.Geometry.Point(lon, lat)
        region_of_interest = point_of_interest.buffer(buffer_m)

        # select the first image, and turn the colleciton into an 8-bit RGB for download
        selectedImage = cloudfree_collection.first()
        mosaic = cloudfree_collection.sort('timediff', False).mosaic()
        rgb = mosaic.select('B4', 'B3', 'B2')
        rgb = rgb.unitScale(0, 15000).clamp(0.0, 1.0)
        rgb_gamma = rgb.pow(1/gamma_value)
        rgb8bit= rgb_gamma.multiply(255).uint8()

        # from the selected image get some stats: product id, cloud probability and time difference from icesat-2
        prod_id = selectedImage.get('PRODUCT_ID').getInfo()
        cld_prb = selectedImage.get('ground_track_cloud_prob').getInfo()
        s2datetime = datetime.fromtimestamp(selectedImage.get('system:time_start').getInfo()/1e3)
        s2datestr = datetime.strftime(s2datetime, '%Y-%b-%d')
        is2datetime = datetime.strptime(date_time, '%Y-%m-%dT%H:%M:%SZ')
        timediff = s2datetime - is2datetime
        days_diff = timediff.days
        if days_diff == 0: diff_str = 'Same day as'
        if days_diff == 1: diff_str = '1 day after'
        if days_diff == -1: diff_str = '1 day before'
        if days_diff > 1: diff_str = '%i days after' % np.abs(days_diff)
        if days_diff < -1: diff_str = '%i days before' % np.abs(days_diff)

        print('--> Closest cloud-free Sentinel-2 image:')
        print('    - product_id: %s' % prod_id)
        print('    - time difference: %s' % timediff)
        print('    - mean cloud probability: %.1f' % cld_prb)

        # get the download URL and download the selected image
        success = False
        scale = 10
        tries = 0
        while (success == False) & (tries <= 5):
            try:
                downloadURL = rgb8bit.getDownloadUrl({'name': 'mySatelliteImage',
                                                          'crs': selectedImage.select('B3').projection().crs(),
                                                          'scale': scale,
                                                          'region': region_of_interest,
                                                          'filePerBand': False,
                                                          'format': 'GEO_TIFF'})

                response = requests.get(downloadURL)
                with open(imagery_filename, 'wb') as f:
                    f.write(response.content)

                print('--> Downloaded the 8-bit RGB image as %s.' % imagery_filename)
                success = True
                tries += 1
            except:
                # traceback.print_exc()
                scale *= 2
                print('-> download unsuccessful, increasing scale to %.1f...' % scale)
                success = False
                tries += 1

    myImage = rio.open(imagery_filename)

    # make the figure
    if not ax:
        fig, ax = plt.subplots(figsize=[6,6])

    rioplot.show(myImage, ax=ax)
    ax.axis('off')
    add_graticule(img=myImage, ax_img=ax)

    if download_imagery:
        # add some info about the Sentinel-2 image
        txt = 'Sentinel-2 on %s\n' % s2datestr
        txt += '%s\n' % prod_id
        txt += '- time difference: %s\n' % timediff
        txt += '- mean cloud probability: %.1f%%' % cld_prb
        ax.text(0.0, -0.01, txt, transform=ax.transAxes, ha='left', va='top',fontsize=6)
    
    if not ax:
        fig.tight_layout()
        fig.savefig(plot_filename, dpi=600)
        print('--> Saved plot as %s.' % plot_filename)
        
    return myImage


#####################################################################
def convert_time_to_string(lake_mean_delta_time):
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


# function to add map inset
#####################################################################
def add_inset(ax, lat, lon, inset, loc=[0.69, 0.01], width=0.3, height=0.25):
    
    if not inset:
        return
    
    bnds = [loc[0], loc[1], width, height]
    axi = ax.inset_axes(bounds=bnds)
    
    if inset.lower().strip() == 'antarctica':
        coast = gpd.read_file('data/shapefiles/Coastline_Antarctica_v02.shp')
        shelf = gpd.read_file('data/shapefiles/IceShelf_Antarctica_v02.shp')
        ground = gpd.read_file('data/shapefiles/GroundingLine_Antarctica_v02.shp')
        tol = 30000
        ground.dissolve().simplify(tolerance=tol).plot(color=[0.95]*3, ax=axi, lw=0.5, alpha=1)
        shelf.dissolve().simplify(tolerance=tol).plot(color=[0.85]*3, ax=axi, lw=0.5, alpha=1)
        ground.boundary.simplify(tolerance=tol).plot(color='k', ax=axi, lw=0.1)
        coast.simplify(tolerance=tol).exterior.plot(color='k', ax=axi, lw=0.3)
        point = gpd.GeoSeries(Point(lon, lat), crs='EPSG:4326').to_crs(coast.crs)
        axi.scatter(point.loc[0].x, point.loc[0].y, s=4, color='r', zorder=1000)
        
    axi.axis('off')
        
    
#####################################################################
def plotIS2S2(lon, lat, date, rgt, gtx, buffer_m=2500, max_cloud_prob=15, gamma_value=1.0, 
              imagery_filename=None, plot_filename=None, title='ICESat-2 ATL03 data',
              ylim=None, IS2dir='IS2data', return_data=False, apply_geoid=True, inset=False, 
              re_download=True):
    
    bbox_atl03 = get_bbox(lon, lat, buffer_m)
    uid, pwd, email = getedcreds()

    if not os.path.exists(IS2dir):
        os.makedirs(IS2dir)
    
    # look for files that have already been downloaded, if re_download is set to False
    atl03file = None
    if not re_download:
        # naming convention: ATL03_[yyyymmdd][hhmmss]_[ttttccss]_[vvv_rr].h5
        search_pattern = r'processed_ATL03_%s\d{6}_%04i\d{4}_\d{3}_\d{2}.h5$' % (date.replace('-', ''), rgt)
        regex = re.compile(search_pattern)
        filename = None
        for root, dirs, files in os.walk(IS2dir):
            for file in files:
                if regex.match(file):
                    print('found already downloaded file: %s/%s' % (root, file))
                    atl03file = '%s/%s' % (root, file)
                    break
    
    # if no matching file was found, or if re-download is set to True, download the data from NSIDC
    if not atl03file:
        granule_list = download_is2(short_name='ATL03', rgt=rgt, start_date=date, end_date=date, 
                                    boundbox=bbox_atl03, output_dir=IS2dir, uid=uid, pwd=pwd)

        atl03file = [IS2dir + '/' + x for x in os.listdir(IS2dir) if granule_list[0] in x][0]
        
    if not imagery_filename:
        imagery_filename = 'imagery/' + atl03file.split('/')[-1].replace('.h5', '') + '_%s' % gtx + '.tif'
        
    # read in the data from h5 file
    gtxs, ancillary, photon_data = read_atl03(atl03file, geoid_h=apply_geoid)
    df = photon_data[gtx]
    date_time = convert_time_to_string(df.dt.median())

    # make the figure with ICESat-2 data
    fig = plt.figure(figsize=[10,4])
    gs = fig.add_gridspec(1, 5)
    ax1 = fig.add_subplot(gs[0, 0:2])
    
    download_imagery = re_download | ( not os.path.exists(imagery_filename) )
    myImage = plotS2cloudfree(lon=lon, lat=lat, date_time=date_time, buffer_m=buffer_m, 
        max_cloud_prob=max_cloud_prob, gamma_value=gamma_value, imagery_filename=imagery_filename, 
        plot_filename='plots/plot.jpg', ax=ax1, download_imagery=download_imagery)

    ax2 = fig.add_subplot(gs[0, 2:])
    df.xatc -= df.xatc.min()
    df['x10'] = np.round(df.xatc, -1)
    gt = df.groupby(by='x10')[['lat', 'lon']].median().reset_index()
    ximg, yimg = warp.transform(src_crs='epsg:4326', dst_crs=myImage.crs, xs=np.array(gt.lon), ys=np.array(gt.lat))
    ax1.annotate('', xy=(ximg[-1], yimg[-1]), xytext=(ximg[0], yimg[0]),
                         arrowprops=dict(width=0.7, headwidth=5, headlength=5, color='r'),zorder=1000)
    ax1.plot(ximg, yimg, 'r-', lw=0.5, zorder=500)

    ax = ax2
    # atl03scatt = ax.scatter(df.lat, df.h, s=1, c='k', label='ATL03 photons', alpha=np.clip(df.weight_ph/100,0,1))
    atl03scatt = ax.scatter(df.lat, df.h, s=1, c='k', label='ATL03 photons', alpha=1)
    ax.set_xlim((df.lat.min(), df.lat.max()))
    if not ylim:
        signal = df.weight_ph > 100
        maxy = df.h[signal].max() 
        miny = df.h[signal].min()
        rngy = maxy-miny
        yl = (miny - rngy, maxy + rngy)
        ax.set_ylim(yl)
    else:
        ax.set_ylim(ylim)
    # adjust font sizes
    ax.tick_params(labelsize=7)
    ax.set_xlabel('', fontsize=8)
    ax.set_ylabel('elevation (m)', fontsize=8)
    ax.legend(handles=[atl03scatt], loc='upper right', fontsize=8, scatterpoints=3)

    # flip x-axis if track is descending, to make along-track distance go from left to right
    if gt.lat.iloc[0] > gt.lat.iloc[-1]:
        ax.set_xlim(np.flip(np.array(ax.get_xlim())))

    # add along-track distance
    lx = gt.sort_values(by='x10').iloc[[0,-1]][['x10','lat']].reset_index(drop=True)
    _lat = np.array(lx.lat)
    _xatc = np.array(lx.x10) / 1e3
    def lat2xatc(l):
        return _xatc[0] + (l - _lat[0]) * (_xatc[1] - _xatc[0]) /(_lat[1] - _lat[0])
    def xatc2lat(x):
        return _lat[0] + (x - _xatc[0]) * (_lat[1] - _lat[0]) / (_xatc[1] - _xatc[0])
    secax = ax.secondary_xaxis(-0.075, functions=(lat2xatc, xatc2lat))
    secax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    secax.set_xlabel('latitude / along-track distance (km)',fontsize=8,labelpad=0)
    secax.tick_params(axis='both', which='major', labelsize=7)
    # secax.ticklabel_format(useOffset=False) # show actual readable latitude values
    secax.ticklabel_format(useOffset=False, style='plain')
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_title(title)

    txt = 'ICESat-2: %s UTC ' % date_time
    txt += '(%g°N, %g°E)\n' % (lat, lon)
    txt += 'RGT %s - %s (%s) - cycle %i: ' % (rgt, gtx.upper(), ancillary['gtx_strength_dict'][gtx], ancillary['cycle_number'])
    txt += '%s' % ancillary['granule_id']
    tbx = ax.text(0.01, 0.02, txt, transform=ax.transAxes, ha='left', va='bottom',fontsize=6)
    tbx.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white', pad=0))
    
    if inset:
        add_inset(fig, lat, lon, inset, loc=[0.69, 0.01], width=0.3, height=0.25)

    fig.tight_layout()

    if not plot_filename:
        plot_filename = 'plots/IS2_cycle%02i_RGT%04i_%s_%s_%s.jpg' % (ancillary['cycle_number'],
                                                            rgt,
                                                            gtx.upper(),
                                                            date_time,
                                                            ancillary['gtx_strength_dict'][gtx]
                                                           )
    fig.savefig(plot_filename, dpi=600)
    print('--> Saved plot as %s.' % plot_filename)
    
    if return_data:
        return fig, df, ancillary, myImage
    else:
        return fig