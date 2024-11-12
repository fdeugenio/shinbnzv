# define 3Mbds variables
import os
import pickle

__all__ = ('preshock_interp', 'shock_interp')

shock__no_pre_file_pckl = 'logshock__no_pre.pckl'
shock_and_pre_file_pckl = 'logshock_and_pre.pckl'

print('Loading shock only...')
if os.path.isfile(shock__no_pre_file_pckl):
    shock_interp = pickle.load(open(shock__no_pre_file_pckl, 'rb'))
    print(f'Loaded interpolator from file {shock__no_pre_file_pckl}')
else:
    raise FileNotFoundError('You need the static files `logshock_[_no,and]_pre.{fits,pckl}')

print('Loading shock+precursor...')
if os.path.isfile(shock_and_pre_file_pckl):
    preshock_interp = pickle.load(open(shock_and_pre_file_pckl, 'rb'))
    print(f'Loaded interpolator from file {shock_and_pre_file_pckl}')
else:
    raise FileNotFoundError('You need the static files `logshock_[_no,and]_pre.{fits,pckl}')
