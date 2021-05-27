#!/usr/bin/env python3

import yaml
import sys
import os
import gc
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.sparse import vstack, save_npz

gc.enable()
logger = logging.getLogger('brane')
logger.setLevel(logging.DEBUG)

DTYPES = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }

#eval_metric: 
def preprocess(use_local: bool, use_sampled_data: bool):
    use_sampled_data_str = '1000' if use_sampled_data else ''
    data_loc_prefix = '../data/' if use_local else '/data/data/'
        
    # logger.info('Loading Train and Test Data into dataframe')
    # print("test print")
    train = pd.read_csv(f"{data_loc_prefix}train{use_sampled_data_str}.csv", dtype=DTYPES)
    train['MachineIdentifier'] = train.index.astype('uint32')
    test  = pd.read_csv(f"{data_loc_prefix}test{use_sampled_data_str}.csv",  dtype=DTYPES)
    test['MachineIdentifier']  = test.index.astype('uint32')
    gc.collect()

    ###
    # logger.info('Transform all features to category')
    for usecol in train.columns.tolist()[1:-1]:

        train[usecol] = train[usecol].astype('str')
        test[usecol] = test[usecol].astype('str')
        # doing this twice seems to fix the nans dont ask me why
        train[usecol] = train[usecol].astype('str')
        test[usecol] = test[usecol].astype('str')

        #Fit LabelEncoder
        le = LabelEncoder().fit(train[usecol].tolist()+test[usecol].tolist())

        #At the end 0 will be used for dropped values
        train[usecol] = le.transform(train[usecol])+1
        test[usecol]  = le.transform(test[usecol])+1

        agg_tr = (train
                .groupby([usecol])
                .aggregate({'MachineIdentifier':'count'})
                .reset_index()
                .rename({'MachineIdentifier':'Train'}, axis=1))
        agg_te = (test
                .groupby([usecol])
                .aggregate({'MachineIdentifier':'count'})
                .reset_index()
                .rename({'MachineIdentifier':'Test'}, axis=1))

        agg = pd.merge(agg_tr, agg_te, on=usecol, how='outer').replace(np.nan, 0)
        #Select values with more than 1000 observations
        agg = agg[(agg['Train'] > 1000)].reset_index(drop=True)
        agg['Total'] = agg['Train'] + agg['Test']
        #Drop unbalanced values
        agg = agg[(agg['Train'] / agg['Total'] > 0.2) & (agg['Train'] / agg['Total'] < 0.8)]
        agg[usecol+'Copy'] = agg[usecol]

        train[usecol] = (pd.merge(train[[usecol]], 
                                agg[[usecol, usecol+'Copy']], 
                                on=usecol, how='left')[usecol+'Copy']
                        .replace(np.nan, 0).astype('int').astype('category'))

        test[usecol]  = (pd.merge(test[[usecol]], 
                                agg[[usecol, usecol+'Copy']], 
                                on=usecol, how='left')[usecol+'Copy']
                        .replace(np.nan, 0).astype('int').astype('category'))

        del le, agg_tr, agg_te, agg, usecol
        gc.collect()
            
    y_train = np.array(train['HasDetections'])
    train_ids = train.index
    test_ids  = test.index
    train_ids.to_series().to_pickle(f'{data_loc_prefix}_train_index{use_sampled_data_str}.pkl')
    test_ids.to_series().to_pickle(f'{data_loc_prefix}_test_index{use_sampled_data_str}.pkl')
    np.save(f'{data_loc_prefix}_train{use_sampled_data_str}.npy', y_train)

    del train['HasDetections'], train['MachineIdentifier'], test['MachineIdentifier']
    gc.collect()

    # logger.info("If you don't want use Sparse Matrix choose Kernel Version 2 to get simple solution.\n")

    # logger.info('--------------------------------------------------------------------------------------------------------')
    # logger.info('Transform Data to Sparse Matrix.')
    # logger.info('Sparse Matrix can be used to fit a lot of models, eg. XGBoost, LightGBM, Random Forest, K-Means and etc.')
    # logger.info('To concatenate Sparse Matrices by column use hstack()')
    # logger.info('Read more about Sparse Matrix https://docs.scipy.org/doc/scipy/reference/sparse.html')
    # logger.info('Good Luck!')
    # logger.info('--------------------------------------------------------------------------------------------------------')

    #Fit OneHotEncoder
    ohe = OneHotEncoder(categories='auto', sparse=True, dtype='uint8').fit(train)

    #Transform data using small groups to reduce memory usage
    m = 100000
    train = vstack([ohe.transform(train[i*m:(i+1)*m]) for i in range(train.shape[0] // m + 1)])
    test  = vstack([ohe.transform(test[i*m:(i+1)*m])  for i in range(test.shape[0] // m +  1)])
    save_npz(f'{data_loc_prefix}_train{use_sampled_data_str}.npz', train, compressed=True)
    save_npz(f'{data_loc_prefix}_test{use_sampled_data_str}.npz',  test,  compressed=True)

    del ohe, train, test
    gc.collect()
    return "Preprocessed data"


if __name__ == "__main__":
    command = sys.argv[1]
    use_local = os.environ["USE_LOCAL"] in ['true', 'True', True]
    use_sampled_data = os.environ["USE_SAMPLED_DATA"] in ['true', 'True', True]
    functions = {
    "preprocess": preprocess,
    }
    output = functions[command](use_local, use_sampled_data)
    print(yaml.dump({"output": output}))
