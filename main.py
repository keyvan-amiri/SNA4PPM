# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 13:26:08 2025
@author: Keyvan Amiri Elyasi
"""
import os
import argparse
import yaml
import logging
from utils.utils import augment_args

# Configure logger
logger = logging.getLogger('SNA4PPM') 
logger.setLevel(logging.INFO) 
# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():    
    # parse arguments and load the relevant cfg file
    parser = argparse.ArgumentParser(description='SNA4PPM')
    parser.add_argument('--dataset') 
    parser.add_argument('--task')
    parser.add_argument('--cfg_file')
    parser.add_argument('--seed', type=int, default=42)   
    args = parser.parse_args() 
    # get the path for root directory on local machine   
    args.root_path = os.getcwd()
    # load the cfg file
    args.cfg_path = os.path.join(args.root_path, 'cfg', args.cfg_file)
    with open(args.cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    args = augment_args(args, cfg)
        
  
    
    
