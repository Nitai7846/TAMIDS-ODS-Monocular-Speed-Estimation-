import json
import os
import shutil
from tqdm import tqdm
import glob
import argparse

from gps_utils import visualize_points
from pano_utils import convert_equirect_perspective
from download import GSVDownloader


locations_to_params = {
    'jhts': {
        'center_lat': 30.624820, 
        'center_lng': -96.433409,
        'radius': 200,
        'resolution': 5,
        'min_year': 2019,
        'max_year': 2024,
        'max_num_panos': 50,
        'max_distance': 100
    },
}

default_params = {
    'raw_data_path': '/Users/nitaishah/Desktop/highway_data/raw',
    'processed_data_path': '/Users/nitaishah/Desktop/highway_data/processed'
}


def download_location(location, api_key):
    # Get params for location
    params = locations_to_params[location]

    print('>>> Downloading data for location {}, with params'.format(location, params))

    # Setup downloader
    gsv_downloader = GSVDownloader(api_key=api_key)

    # Data path
    root_data_path = os.path.join(default_params['raw_data_path'], location)
    if not os.path.exists(root_data_path):
        os.makedirs(root_data_path)

    # # Query panos
    print('Querying panos for location {}'.format(location))
    output_json_path = os.path.join(root_data_path, 'queried_panos.json')
    gsv_downloader.query_panos(center_lat=params['center_lat'], 
                              center_lng=params['center_lng'],
                              radius=params['radius'], 
                              resolution=params['resolution'], 
                              do_viz=False,
                              output_json_path=output_json_path)
    
    # Load the queried panos
    with open(output_json_path, 'r') as f:
        pano_infos = json.load(f)

    # Filter panos
    print('Filtering panos for location {}'.format(location))
    pruned_panos_json_path = os.path.join(root_data_path, 'pruned_panos.json')
    gsv_downloader.filter_panos(pano_infos=pano_infos,
                                origin_gps=(params['center_lat'], params['center_lng']),
                                min_year=params['min_year'], 
                                max_year=params['max_year'], 
                                max_distance=params['max_distance'],
                                max_num_panos=params['max_num_panos'],
                                pruned_panos_json_path=pruned_panos_json_path)

    # Load the pruned panos
    with open(pruned_panos_json_path, 'r') as f:
        pruned_panos = json.load(f)

    # Visualize the pruned panos to double check
    pano_gps = [(pano['lat'], pano['lon']) for pano in pruned_panos]
    visualize_points(pano_gps)
        
    # Download panos
    print('Downloading panos for location {}'.format(location))
    output_dir = os.path.join(root_data_path, 'pano_data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cnt = 0
    for pano_info in pruned_panos:
        panoid = pano_info['panoid']
        status = gsv_downloader.download_pano(panoid=panoid,
                                              zoom=4,
                                              output_filename="{:06d}_{}.{}".format(cnt, panoid, 'png'),
                                              output_dir=output_dir)
        if status:
            print('Successfully downloaded pano {}'.format(panoid))
            cnt += 1
        else:
            print('Failed to download pano {}'.format(panoid))

    

def process_location(location):
    pano_data_dir = os.path.join(default_params['raw_data_path'], location, 'pano_data')
    perspective_output_data_dir = os.path.join(default_params['raw_data_path'], location, 'perspective_data')
    database_output_data_dir = os.path.join(default_params['processed_data_path'], location, 'database/images')
    query_data_dir = os.path.join(default_params['processed_data_path'], location, 'query/images')
    os.makedirs(database_output_data_dir, exist_ok=True)
    os.makedirs(query_data_dir, exist_ok=True)
    
    # Extract perspective images
    pano_names = sorted(os.listdir(pano_data_dir))
    for pano_name in tqdm(pano_names):
        print('Converting {} to perspective'.format(pano_name))
        convert_equirect_perspective(pano_dir=pano_data_dir, 
                                     pano_name=os.path.splitext(pano_name)[0], 
                                     output_dir=perspective_output_data_dir)

    # Copy perspective images to database to prepare for COLMAP
    for filepath in glob.glob(os.path.join(perspective_output_data_dir, '**/*.png')):
        newpath = os.path.join(database_output_data_dir, os.path.basename(filepath))
        shutil.copy(filepath, newpath)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default='jhts')
    parser.add_argument('--api_key', type=str, required=True)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    # Downloading data
    download_location(args.location, args.api_key)

    # Processing data
    process_location(args.location)

    


