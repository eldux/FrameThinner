"""Thin visually redundant frames from one video into a separate output folder."""
import argparse
from collections import Counter
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
from shutil import copy2
import sys
from time import perf_counter

import cv2
import numpy as np
import yaml

from annotations import annotations_match, read_annotations
from motion import features, motion

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def natural_key(path):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r'(\d+)', path.name))


def load_config(path):
    with path.open(encoding='utf-8-sig') as stream:
        config = yaml.safe_load(stream)
    with (ROOT/'config.yaml').open(encoding='utf-8-sig') as stream:
        defaults = yaml.safe_load(stream)
    def merge(base, overrides):
        if not isinstance(overrides, dict):
            raise ValueError('Configuration sections must be mappings.')
        for key, value in overrides.items():
            if key not in base:
                raise ValueError(f'Unknown configuration key: {key}')
            if isinstance(base[key], dict):
                merge(base[key], value)
            else:
                base[key] = value
    merge(defaults, config)
    c = defaults
    if type(c['maximum_motion']) not in (int, float) or not math.isfinite(c['maximum_motion']) or c['maximum_motion'] <= 0:
        raise ValueError('maximum_motion must be finite and positive.')
    for section, names in (('features', ('orb_count','orb_fast_threshold','sift_count')),
                           ('quality', ('min_matches','min_inliers')),
                           ('runtime', ('cpu_threads','feature_cache_frames'))):
        for name in names:
            if type(c[section][name]) is not int or c[section][name] < 1:
                raise ValueError(f'{section}.{name} must be a positive integer.')
    for section, name in (('quality','match_ratio'),('quality','min_inlier_ratio'),
                           ('vehicles','confidence'),('features','sift_contrast')):
        value = c[section][name]
        if type(value) not in (float,int) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f'{section}.{name} must be in (0,1].')
    for section, name in ((None,'include_sky'),(None,'repeat_until_stable'),
                          ('features','sift_fallback'),('vehicles','enabled')):
        value = c[section][name] if section else c[name]
        if type(value) is not bool:
            raise ValueError(f'{name} must be true or false.')
    if type(c['runtime']['periodic_interval']) is not int or c['runtime']['periodic_interval'] < 0:
        raise ValueError('periodic_interval must be a nonnegative integer.')
    for section, name in (('features','sift_scale'),('quality','ransac_pixels')):
        value = c[section][name]
        if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{section}.{name} must be finite and positive.')
    if (not isinstance(c['quality']['min_spread_pixels'], list)
            or len(c['quality']['min_spread_pixels']) != 2
            or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0
                   for v in c['quality']['min_spread_pixels'])):
        raise ValueError('min_spread_pixels must contain two nonnegative numbers.')
    if type(c['regions']['vehicle_padding']) is not int or c['regions']['vehicle_padding'] < 0:
        raise ValueError('vehicle_padding must be a nonnegative integer.')
    if not c['vehicles']['classes'] or any(type(v) is not int or v < 0 for v in c['vehicles']['classes']):
        raise ValueError('Vehicle classes must be nonnegative class IDs.')
    regions = c['regions']
    for name in ('road_polygon',):
        if len(regions[name]) < 3 or any(len(p)!=2 or not 0 <= p[0] <= 640 or not 0 <= p[1] <= 480 for p in regions[name]):
            raise ValueError('Invalid road polygon.')
    x1,y1,x2,y2 = regions['upper_rectangle']
    if not 0 <= x1 < x2 <= 640 or not 0 <= y1 < y2 <= 480:
        raise ValueError('Invalid upper_rectangle.')
    for name, size in (('grid_x',640),('road_grid_y',480),('upper_grid_y',480)):
        if not regions[name] or any(not 0 <= x <= size for x in regions[name]):
            raise ValueError(f'Invalid {name}')
    return c


def file_state(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


class FrameComparator:
    def __init__(self, paths, config, config_path):
        self.paths = paths
        self.config = config
        cv2.setNumThreads(config['runtime']['cpu_threads'])
        opts = config['features']
        self.detectors = {
            False: cv2.ORB_create(nfeatures=opts['orb_count'], fastThreshold=opts['orb_fast_threshold']),
            True: cv2.SIFT_create(nfeatures=opts['sift_count'], contrastThreshold=opts['sift_contrast']),
        }
        self.matchers = {False:cv2.BFMatcher(cv2.NORM_HAMMING), True:cv2.BFMatcher(cv2.NORM_L2)}
        self.model = None
        self.device = 'cpu'
        if config['vehicles']['enabled']:
            os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'.runtime'/'ultralytics'))
            os.environ.setdefault('YOLO_AUTOINSTALL','false')
            import torch
            from ultralytics import YOLO
            weights = (config_path.parent / config['vehicles']['weights']).resolve()
            if not weights.is_file():
                raise FileNotFoundError(f'Vehicle weights missing: {weights}')
            os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'.runtime'/'ultralytics'))
            os.environ.setdefault('YOLO_AUTOINSTALL','false')
            self.device = config['vehicles']['device']
            if self.device == 'auto':
                self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.model = YOLO(str(weights), task='detect')
        size = config['runtime']['feature_cache_frames']
        self.boxes = lru_cache(maxsize=size)(self._boxes)
        self.describe = lru_cache(maxsize=size*4)(self._describe)
        self.evidence = lru_cache(maxsize=2048)(self._evidence)

    def _boxes(self, index):
        if self.model is None:
            return ()
        pixels = cv2.imdecode(np.fromfile(self.paths[index],dtype=np.uint8),cv2.IMREAD_REDUCED_COLOR_2)
        if pixels is None:
            raise ValueError(f'Cannot decode {self.paths[index]}')
        pixels = cv2.resize(pixels,(640,480),interpolation=cv2.INTER_AREA)
        try:
            result = self.model.predict(pixels, imgsz=640, conf=self.config['vehicles']['confidence'],
                                        classes=self.config['vehicles']['classes'], device=self.device, verbose=False)[0]
            return tuple(map(tuple,result.boxes.xyxy.cpu().tolist()))
        finally:
            if self.model.predictor:
                self.model.predictor.batch = None
                self.model.predictor.results = None
                self.model.predictor.dataset = None

    def _describe(self, index, upper, dense):
        scale = self.config['features']['sift_scale'] if dense else 1
        return features(self.paths[index], self.detectors[dense], upper_scene=upper,
                        excluded_boxes=self.boxes(index), scale=scale,
                        mask_sky=not self.config['include_sky'], regions=self.config['regions'])

    def _evidence(self, left, right, upper):
        def compare(dense):
            return motion(self.describe(left,upper,dense),self.describe(right,upper,dense),
                          self.matchers[dense],self.config['quality'],self.config['regions'],upper_scene=upper)
        regular = compare(False)
        if 'score' in regular or not self.config['features']['sift_fallback']:
            return regular
        return {**compare(True), 'method':'SIFT fallback', 'orb':regular}

    def decide(self, left, right):
        threshold = self.config['maximum_motion']
        road = self.evidence(left,right,False)
        row = {**road,'status':'retain','reference':self.paths[left].name}
        if 'score' not in road:
            return {**row,'status':'uncertain'}
        if road['score'] >= threshold or road['horizontal'] >= threshold or road['peak'] >= 2*threshold:
            return {**row,'reason':'Movement reaches a retention limit.'}
        upper = self.evidence(left,right,True)
        row['upper'] = upper
        if 'peak' not in upper:
            return {**row,'status':'uncertain','reason':'Upper scene alignment is insufficient.'}
        if upper['peak'] >= 2*threshold:
            return {**row,'reason':'Upper scene movement reaches the retention limit.'}
        return {**row,'status':'remove_candidate','reason':'All movement checks indicate a redundant frame.'}


def run(args, frame_paths=None, comparator_factory=FrameComparator):
    source, output = args.input.resolve(), args.output.resolve()
    if not source.is_dir():
        raise ValueError('Input must be a folder containing frames from one video.')
    if output==source or output in source.parents or source in output.parents:
        raise ValueError('Input and output must be separate, non-nested folders.')
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    paths = sorted((p for p in source.iterdir() if p.is_file() and not p.is_symlink()
                    and p.suffix.lower() in EXTENSIONS), key=lambda p:(natural_key(p),p.name))
    if frame_paths is not None:
        allowed=set(paths)
        requested=list(frame_paths)
        if not requested or len(set(requested)) != len(requested) or not set(requested).issubset(allowed):
            raise ValueError('Selected frames must be unique files directly inside the input folder.')
        paths=sorted(requested,key=lambda p:(natural_key(p),p.name))
    if not paths:
        raise ValueError('No supported images directly inside input.')
    labels = None
    annotations = None
    if args.labels:
        label_root=args.labels.resolve()
        if output==label_root or output in label_root.parents or label_root in output.parents:
            raise ValueError('Output must be separate from the labels folder.')
        if len({p.stem.casefold() for p in paths}) != len(paths):
            raise ValueError('Image stems must be unique when copying labels.')
        labels=[label_root/(p.stem+'.txt') for p in paths]
        if any(not p.is_file() for p in labels):
            raise ValueError('Every input image must have a matching YOLO .txt file.')
        annotations=[read_annotations(p) for p in labels]
    config_path=args.config.resolve()
    config=load_config(config_path)
    states={p:file_state(p) for p in paths+(labels or [])}
    comparator=comparator_factory(paths,config,config_path)
    output.mkdir(parents=True)
    started=perf_counter()
    survivors=list(range(len(paths)))
    passes=[]
    compared=0
    while True:
        anchor=survivors[0]
        remaining=[anchor]
        decisions=[{'index':anchor,'image':paths[anchor].name,'status':'retain','reason':'Run endpoint.'}]
        for position,index in enumerate(survivors[1:],1):
            row=comparator.decide(anchor,index)
            if row['status']=='remove_candidate':
                if index==len(paths)-1:
                    row.update(status='retain',reason='Run endpoint.')
                elif config['runtime']['periodic_interval'] and index-anchor>=config['runtime']['periodic_interval']:
                    row.update(status='retain',reason='Periodic sample.')
                elif annotations is not None and not annotations_match(annotations[anchor],annotations[index]):
                    row.update(status='retain',reason='Defect annotations materially differ.')
            row.update(index=index,image=paths[index].name)
            decisions.append(row)
            if row['status']!='remove_candidate':
                remaining.append(index)
                anchor=index
            compared+=1
            rate=compared/max(.001,perf_counter()-started)
            eta=(len(survivors)-position-1)/max(.001,rate)
            if position%10==0 or position==len(survivors)-1:
                print(f'Pass {len(passes)+1} | {position+1}/{len(survivors)} frames | '
                      f'{len(remaining)} retained so far | {rate:.2f} comparisons/s | '
                      f'pass ETA {eta:.0f}s',flush=True)
        removed=len(survivors)-len(remaining)
        passes.append({'number':len(passes)+1,'removed':removed,'retained':remaining,'decisions':decisions})
        print(f'Pass {len(passes)} complete: {removed} removal candidates, {len(remaining)} retained.',flush=True)
        survivors=remaining
        if not removed or not config['repeat_until_stable']:
            break
    if any(file_state(p)!=snapshot for p,snapshot in states.items()):
        raise ValueError('Input changed while processing. Rerun from a new output folder.')
    report={'input':str(source),'labels':str(args.labels.resolve()) if args.labels else None,
            'configuration':config,'images':len(paths),'retained':len(survivors),
            'removed_candidates':len(paths)-len(survivors),'passes':passes,
            'elapsed_seconds':round(perf_counter()-started,2),'dry_run':args.dry_run}
    if not args.dry_run:
        for index in survivors:
            destination=output/'images'/paths[index].name
            destination.parent.mkdir(exist_ok=True)
            copy2(paths[index],destination)
            if labels:
                destination=output/'labels'/labels[index].name
                destination.parent.mkdir(exist_ok=True)
                copy2(labels[index],destination)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Finished: {len(survivors)}/{len(paths)} retained. Report: {output/"report.json"}',flush=True)
    comparator.evidence.cache_clear()
    comparator.describe.cache_clear()
    comparator.boxes.cache_clear()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--labels',type=Path,help='Optional parallel folder of YOLO labels')
    parser.add_argument('--config',type=Path,default=ROOT/'config.yaml')
    parser.add_argument('--dry-run',action='store_true',help='Write the decision report only')
    args=parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        parser.exit(130,'Stopped. Use a new output directory to rerun.\n')
    except Exception as error:
        parser.exit(1,f'FrameThinner stopped: {error}\n')


if __name__=='__main__':
    main()
