import numpy as np
from functools import lru_cache

__version__ = '0.6.5'

recordType = [
    ('reserved', 'bool'), 
    ('valid',    'bool'), 
    ('routing',  'uint8'),
    ('channel',  'uint16'), 
    ('timetag',  'uint64')
]

# headerType currently assumes a static header exactly formatted as described. 
# There is no dynamic evaluation of the files shape.
headerType = [
    ('Ident', 'S16'),
    ('FormatVersion', 'S6'),
    ('CreateorName', 'S18'),
    ('CreatorVersion', 'S12'),
    ('FileTime', 'S18'),
    ('CR/LF', 'S2'),
    ('Comment', 'S256'),
    ('NumberOfChannels', 'int32'), # changes dynamically
    ('NumberOfCurves', 'int32'),   # changes dynamically
    ('BitsPerChannael', 'int32'),   # changes dynamically
    ('RoutingChannels', 'int32'),
    ('NumberOfBoards', 'int32'),   # changes dynamically
    ('ActiveCurve', 'int32'),
    ('MeasuermentMode', 'int32'),  # changes dynamically
    ('SubMode', 'int32'),
    ('RangeNo', 'int32'),
    ('Offset', 'int32'),
    ('AcquisitionTime', 'int32'),
    ('StopAt', 'int32'),
    ('StopOnOvfl', 'int32'),
    ('Restart', 'int32'),
    ('DisplayLinLog', 'int32'),
    ('DisplayTimeAxisFrom', 'int32'),
    ('DisplayTimeAxisTo', 'int32'),
    ('DisplayCountAxisFrom', 'int32'),
    ('DisplayCountAxisTo', 'int32'),
    ('DisplayCurve[1].MapTo', 'int32'),
    ('DisplayCurve[1].Show', 'int32'),
    ('DisplayCurve[2].MapTo', 'int32'),
    ('DisplayCurve[2].Show', 'int32'),
    ('DisplayCurve[3].MapTo', 'int32'),
    ('DisplayCurve[3].Show', 'int32'),
    ('DisplayCurve[4].MapTo', 'int32'),
    ('DisplayCurve[4].Show', 'int32'),
    ('DisplayCurve[5].MapTo', 'int32'),
    ('DisplayCurve[5].Show', 'int32'),
    ('DisplayCurve[6].MapTo', 'int32'),
    ('DisplayCurve[6].Show', 'int32'),
    ('DisplayCurve[7].MapTo', 'int32'),
    ('DisplayCurve[7].Show', 'int32'),
    ('DisplayCurve[8].MapTo', 'int32'),
    ('DisplayCurve[8].Show', 'int32'),
    ('Param[1].Start', 'float32'),
    ('Param[1].Step', 'float32'),
    ('Param[1].End', 'float32'),
    ('Param[2].Start', 'float32'),
    ('Param[2].Step', 'float32'),
    ('Param[2].End', 'float32'),
    ('Param[3].Start', 'float32'),
    ('Param[3].Step', 'float32'),
    ('Param[3].End', 'float32'),
    ('RepeatMode', 'int32'),
    ('RepeatsPerCurve', 'int32'),
    ('RepeatTime', 'int32'),
    ('RepeatWaitTime', 'int32'),
    ('ScriptName', 'S20'),
    ('HardwareIdent', 'S16'),
    ('HardwareVersion', 'S8'),
    ('BoardSerial', 'int32'),
    ('CFDZeroCross', 'int32'),
    ('CFDDiscriminatorMin', 'int32'),
    ('SYNCLevel', 'int32'),
    ('CurveOffset', 'int32'),
    ('Resolution', 'float32'),
    ('TTTRGlobclock', 'int32'),
    ('ExtDevices', 'int32'),
    ('Reserved001', 'int32'),
    ('Reserved002', 'int32'),
    ('Reserved003', 'int32'),
    ('Reserved004', 'int32'),
    ('Reserved005', 'int32'),
    ('SyncRate', 'int32'),
    ('AverageCFDRate', 'int32'),
    ('StopAfter', 'int32'),
    ('StopReason', 'int32'), 
    ('NumberOfRecords', 'int32'),   # changes dynamically
    ('SpecHeaderLength', 'int32'),  # changes dynamically
    ('Reserved006', 'int32')        # changes dynamically 
]

def unpackt3r(record):

    '''
    Unpack a t3r 32-bit unsigned integer to individual values. The raw data irrespective of
    rollover events are returned. The timetag information is stored in 64-bit to allow addition
    of rollover time to calculate the truetime in place.
    
    Parameters
    ----------
    record : int (numpy.uint32)
        t3r record to unpack

    Returns
    -------
    reserved : boolean
        reserved bit of the t3r record

    valid : boolean
        validity information bit of the t3r record

    routing : numpy.uint8
        routing information, i.e. detection channel (e.g. donor or acceptor)

    channel : numpy.uint16
        nanotime bin

    timetag : numpy.uint64
        timetag / pulse count
    '''
    
    reserved =          ((record & 0x80000000) >> 31) == 1
    valid    =          ((record & 0x40000000) >> 30) == 1
    routing  = np.uint8 ((record & 0x30000000) >> 28)
    channel  = np.uint16((record & 0x0fff0000) >> 16)
    timetag  = np.uint64 (record & 0x0000ffff)
    
    return reserved, valid, routing, channel, timetag

def parse_header(header, jsonify_header=False, **kwargs):
    '''
    Convert a numpy representation of a t3r header into a dictionary.

    Parameters
    ----------
    header : numpy.array(shape(1,), dtype=headerType)
        Single entry numpy array with a headerType imported from a .t3r file.

    jsonify_header : boolean (default False)
        If True, converts all byte-type and numeric entries to strings for json-compatibility.

    **kwargs :
        kwargs can be used to overwrite arbitrary entries of the dictionary.
        Example: To overwrite the SyncRate use: parse_header(header, SyncRate=1e+7)

    Returns
    -------
    header_dict : dictionary
        Dictionary with all header entries stored under their t3r field names as keys.
    '''
    
    header_dict = dict()
    
    for k, v in [(t[0], v) for t, v in zip(headerType, header[0])]:
        
        if k in kwargs:
            print('Override {} with {}. (Was {})'.format(k, kwargs[k], v))
            v = kwargs[k]
        
        if jsonify_header:
            if type(v) == np.bytes_:
                v = v.decode()
            elif type(v) in (np.int32, np.float32):
                v = str(v)

        header_dict.update({k: v})
        
    return header_dict
                  
def load_t3r(path, n=-1, invert_channels=True, jsonify_header=False, silent=False, **kwargs):
    '''
    Import a .t3r file.

    Parameters
    ----------
    path : string
        Filepath to the .t3r file.

    n : int (default -1)
        Number of records. If -1, all records are imported.

    invert_channels : boolean (default True)
        If true, invert the channel (nanotime) information left to right.

    jsonify_header : boolean (default False)
        Pass-through to the parse_header function.
        If True, converts all byte-type and numeric entries to strings for json-compatibility.

    silent : boolean (default False)
        If True, no text output will be printed.

    **kwargs :
        Pass-through to the parse_header function.
        kwargs can be used to overwrite arbitrary entries of the dictionary.
        Example: To overwrite the SyncRate use: parse_header(header, SyncRate=1e+7)

    Returns
    -------
    header : dict
        Header dictionary containig all metadata of the .t3r file. See also the parse_header function.

    data : numpy.array
        Data array containing the unpacked .t3r records as named fields in a structured numpy array.
    '''

    # load header and records as uint32
    with open(path, 'rb') as f:
        header = np.fromfile(f, dtype=headerType, count=1)
        data = np.fromfile(f, dtype='uint32', count=n)
        
    # process header to dictionary
    header = parse_header(header, jsonify_header, **kwargs)
    
    # unpack unit32 int records to unpacked records with rollover
    records = unpackt3r(data)
    size = len(records[0])
    del data

    # allocate struct-array for unpacked records
    struct_array = np.zeros((size,), dtype=recordType)
    
    # pass record data int struct-array
    struct_array['reserved'] = records[0]
    struct_array['valid']    = records[1]
    struct_array['routing']  = records[2]
    struct_array['channel']  = records[3]
    struct_array['timetag']  = records[4]
    
    # create mask for rollover records
    rollover_mask = (struct_array['valid'] == 0) & (struct_array['channel'] == 2048)
    
    # get rollover indeces
    rollover_index = np.where(rollover_mask == 1)[0]

    #
    # create rollover addition array from rollover indices: 
    #
    #            1st rollover                       2nd rollover
    #                 *                                  *
    # [0, 0, 0, 0, 0, 65536, 65536, 65536, 65536, 65536, 131072, 131072, 131072, ...]
    #
    if rollover_index.size > 0:

        rollover = np.uint64(np.digitize(np.arange(0, size), bins=rollover_index)*2**16)

        # apply rollover correction
        struct_array['timetag'] += rollover

        # mask output to true data, discard rollover records
        struct_array = struct_array[np.invert(rollover_mask)]

    data = struct_array

    # invert channel data
    if invert_channels:
        data['channel'] = np.invert(data['channel']) & 0x0fff
    
    if not silent:
        print('...{:64}:\t{:12d} photons loaded.'.format(path[-64:], data.size))

    return header, data


def make_FRETBursts_object(header, data, routing=(2,1), fname='default'):

    '''
    Factory function to assemble a fretbursts object from header and data information of a t3r file.

    Parameters
    ----------
    header : dict
        Dictionary containing the header metadata of .t3r file.
        Mandatory elements are: 'NumberOfChannels' and 'SyncRate' 

    data : numpy.array
        Array containig the t3r tags in named fields.

    routing : tuple (default (2,1))
        Routing identitiy of the donor and acceptor channels as (D, A)

    fname : string (default 'default')
        Filename to store as the origin of the dataset.
    
    Returns
    -------
    d : fretbursts.Data object
    '''

    try:
        import fretbursts as fb
    except ImportError:
        print('Feature required fretbursts to be installed')
        return None
    
    d = fb.Data(
        ph_times_t = [np.array(data['timetag'], dtype='int64')],
        A_em = [np.array(data['routing'] == 1)],
        D_em = [np.array(data['routing'] == 2)],
        lifetime = True, 
        nanotimes_t = [np.array(data['channel'])],
        nanotimes_params= [{  'tcspc_num_bins': header['NumberOfChannels'],
                              'tcspc_unit': header['Resolution']*1e-9}],
        det_donor_accept = [(2,1)],
        det_t = [np.array(data['routing'])],
        clk_p = 1/header['SyncRate'],
        meas_type=['PAX'],
        alternated=True,
        nch = 1,
        fname = fname
    )
    return d


def inspect_t3r(path):

    try:
        import pandas as pd
        from ipywidgets import interact
    except ImportError:
        print('Feature requires pandas and ipywidgets to be intalled')
        return None

    header, data = load_t3r(path, silent=True)
    pd.set_option('display.max_rows', 50)
    df = pd.DataFrame(data)

    def _inspect_t3r_(i0, entries):
        return df[i0:i0+entries]

    return interact(_inspect_t3r_, i0=(0, data.size), entries=(0,50))

def load_t3r_multi(*paths):
    '''TODO: Cached loading and packing of multiple files.'''
    raise NotImplementedError

@lru_cache(16)
def load_t3r_to_fretbursts_cached(path):
    header, data = load_t3r(path)
    exp = make_FRETBursts_object(header, data)
    return exp
