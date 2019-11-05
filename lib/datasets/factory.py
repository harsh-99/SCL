# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_water import pascal_voc_water
from datasets.pascal_voc_cyclewater import pascal_voc_cyclewater
from datasets.pascal_voc_cycleclipart import pascal_voc_cycleclipart
from datasets.sim10k import sim10k
from datasets.water import water
from datasets.clipart import clipart
from datasets.sim10k_cycle import sim10k_cycle
from datasets.cityscape import cityscape
from datasets.cityscape_car import cityscape_car
from datasets.foggy_cityscape import foggy_cityscape

from datasets.init_sunny import init_sunny
from datasets.init_night import init_night
from datasets.init_rainy import init_rainy
from datasets.init_cloudy import init_cloudy
from datasets.kitti_car import kitti_car

import numpy as np
for split in ['train', 'trainval','val','test']:
  name = 'cityscape_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape(split))
for split in ['train', 'trainval','val','test']:
  name = 'cityscape_car_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape_car(split))
for split in ['train', 'trainval','test', 'val']:
  name = 'foggy_cityscape_{}'.format(split)
  __sets[name] = (lambda split=split : foggy_cityscape(split))
for split in ['train','val']:
  name = 'sim10k_{}'.format(split)
  __sets[name] = (lambda split=split : sim10k(split))

##############################################
for split in ['trainval']:
  name = 'init_sunny_{}'.format(split)
  __sets[name] = (lambda split=split : init_sunny(split))
for split in ['train','val']:
  name = 'init_night_{}'.format(split)
  __sets[name] = (lambda split=split : init_night(split))
for split in ['train','val']:
  name = 'init_rainy_{}'.format(split)
  __sets[name] = (lambda split=split : init_rainy(split))
for split in ['train','val']:
  name = 'init_cloudy_{}'.format(split)
  __sets[name] = (lambda split=split : init_cloudy(split))

for split in ['train']:
  name = 'kitti_car_{}'.format(split)
  __sets[name] = (lambda split=split : kitti_car(split))
##############################################

for split in ['train']:
  name = 'sim10k_cycle_{}'.format(split)
  __sets[name] = (lambda split=split: sim10k_cycle(split))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_water_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc_water(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
      name = 'voc_cycleclipart_{}_{}'.format(year, split)
      __sets[name] = (lambda split=split, year=year: pascal_voc_cycleclipart(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
      name = 'voc_cyclewater_{}_{}'.format(year, split)
      __sets[name] = (lambda split=split, year=year: pascal_voc_cyclewater(split, year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'clipart_{}'.format(split)
    __sets[name] = (lambda split=split : clipart(split,year))
for year in ['2007']:
  for split in ['train', 'test']:
    name = 'water_{}'.format(split)
    __sets[name] = (lambda split=split : water(split,year))
def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
