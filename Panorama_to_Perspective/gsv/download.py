import itertools
import folium
from tqdm import tqdm
import numpy as np
import webbrowser
import requests
import re
from datetime import datetime
import json
from geopy import distance
import os
from PIL import Image
from io import BytesIO
from gps_utils import create_grid, visualize_points



class GSVDownloader:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session_token = self.get_session_key()


    def get_session_key(self):
        url = f"https://tile.googleapis.com/v1/createSession?key={self.api_key}"

        headers = {
            "Content-Type": "application/json",
        }

        data = {
            "mapType": "streetview",
            "language": "en-US",
            "region": "US",
        }

        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            print("Request session key successful!")
            print("Response:")
            print(response.json())
            session_key = response.json()["session"]
            return session_key
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response:")
            print(response.text)
            raise Exception("Failed to get session key")


    def get_closest_panos(self, lat, lng):
        """
        Get the closest panos around a lat/lng
        """
        url = f"https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{lat}!4d{lng}!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
        resp = requests.get(url, proxies=None)
        text = resp.text

        # Get all the panorama ids and coordinates
        # I think the latest panorama should be the first one. And the previous
        # successive ones ought to be in reverse order from bottom to top. The final
        # images don't seem to correspond to a particular year. So if there is one
        # image per year I expect them to be orded like:
        # 2015
        # XXXX
        # XXXX
        # 2012
        # 2013
        # 2014
        pans = re.findall('\[[0-9]+,"(.+?)"\].+?\[\[null,null,(-?[0-9]+.[0-9]+),(-?[0-9]+.[0-9]+)', text)
        pans = [{
            "panoid": p[0],
            "lat": float(p[1]),
            "lon": float(p[2])} for p in pans]  # Convert to floats

        # Remove duplicate panoramas
        pans = [p for i, p in enumerate(pans) if p not in pans[:i]]

        # Get all the dates
        # The dates seem to be at the end of the file. They have a strange format but
        # are in the same order as the panoids except that the latest date is last
        # instead of first.
        dates = re.findall('([0-9]?[0-9]?[0-9])?,?\[(20[0-9][0-9]),([0-9]+)\]', text)
        dates = [list(d)[1:] for d in dates]  # Convert to lists and drop the index

        if len(dates) > 0:
            # Convert all values to integers
            dates = [[int(v) for v in d] for d in dates]

            # Make sure the month value is between 1-12
            dates = [d for d in dates if d[1] <= 12 and d[1] >= 1]

            # The last date belongs to the first panorama
            year, month = dates.pop(-1)
            pans[0].update({'year': year, "month": month})

            # The dates then apply in reverse order to the bottom panoramas
            dates.reverse()
            for i, (year, month) in enumerate(dates):
                pans[-1-i].update({'year': year, "month": month})

        # Sort the pans array
        def func(x):
            if 'year'in x:
                return datetime(year=x['year'], month=x['month'], day=1)
            else:
                return datetime(year=3000, month=1, day=1)
        pans.sort(key=func)

        # Only take the ones with dates
        pans = [p for p in pans if 'year' in p]

        # For this lat-lon, select 1 random panorama (all of them are close to each other, with different dates)
        # the last one is the latest date
        if len(pans) > 0:
            # panoids = [random.choice(panoids)]
            pans = [pans[-1]] # the last one is the latest date

        return pans
    

    # def get_metadata(self, panoid, lat=None, lng=None):
    #     assert panoid is not None or (lat is not None and lng is not None)
    #     if panoid is not None:
    #         url = 'https://maps.googleapis.com/maps/api/streetview/metadata?pano={0}&key={1}'.format(panoid, self.api_key)
    #     else:
    #         url = 'https://maps.googleapis.com/maps/api/streetview/metadata?location={0}%2C{1}&key={2}'.format(lat, lng, self.api_key)
    #     return requests.get(url, proxies=None)
    

    def get_streetview_metadata(self, pano_id):
        url = f"https://tile.googleapis.com/v1/streetview/metadata?session={self.session_token}&key={self.api_key}&panoId={pano_id}"

        response = requests.get(url)

        if response.status_code == 200:
            # print("Request successful!")
            return response.json()
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response:")
            print(response.text)
            return None    


    def query_panos(self, 
                    center_lat: float, 
                    center_lng: float, 
                    radius: float, 
                    resolution: int = 5, 
                    do_viz: bool = False,
                    output_json_path: str = None):
        # Create a grid
        top_left, bottom_right, grid_points = create_grid(center_lat, center_lng, radius, resolution)

        # Visualize the grid
        if do_viz:
            visualize_points(grid_points)

        # Get the closest panos for each grid point
        pano_infos = []
        for point in tqdm(grid_points):
            closest_panos = self.get_closest_panos(point[0], point[1])
            for pano in closest_panos:
                pano_infos.append(pano)

        # De-duplicate the panos by panoid
        deduped_pano_infos = []
        for pano_info in pano_infos:
            if pano_info not in deduped_pano_infos:
                deduped_pano_infos.append(pano_info)
            
        pano_infos = deduped_pano_infos

        # Save the data to a json file
        if output_json_path:
            print(f"Saving data to {output_json_path}")
            with open(output_json_path, 'w') as f:
                json.dump(pano_infos, f, indent=4)

    
    def filter_panos(self, 
                     origin_gps: tuple,
                     pano_infos: list, 
                     min_year: int, 
                     max_year: int,
                     max_distance: float,
                     max_num_panos: int,
                     pruned_panos_json_path: str):
        
        final_panoids = []
        for i, pano_info in enumerate(tqdm(pano_infos)):
            print('>>> pano_info', pano_info)

            if pano_info['year'] < min_year or pano_info['year'] > max_year:
                continue

            # prune by distance
            pano_lat_lng = (pano_info['lat'], pano_info['lon'])
            dist_to_origin = distance.distance(pano_lat_lng, origin_gps).m
            if dist_to_origin > max_distance:
                continue

            pano_info['dist2origin'] = dist_to_origin

            final_panoids.append(pano_info)
        
        final_panoids = sorted(final_panoids, key=lambda i: i['dist2origin'])

        if len(final_panoids) > max_num_panos:
            print('>>>>>> Subsample panolist to get it down to {} panos <<<<<<'.format(max_num_panos))
            final_panoids = np.random.choice(final_panoids, max_num_panos, replace=False).tolist()

        final_panoids = sorted(final_panoids, key=lambda i: i['dist2origin'])

        with open(pruned_panos_json_path, 'w') as file:
            file.write(json.dumps(final_panoids, indent=4))


    def panoid_to_depthinfo(self, panoid):
        # URL of the json file of a GSV depth map
        URL_STR = 'https://www.google.com/maps/photometa/v1?authuser=0&hl=en&gl=uk&pb=!1m4!1smaps_sv.tactile!11m2!2m1!1b1!2m2!1sen!2suk!3m3!1m2!1e2!2s{}!4m57!1e1!1e2!1e3!1e4!1e5!1e6!1e8!1e12!2m1!1e1!4m1!1i48!5m1!1e1!5m1!1e2!6m1!1e1!6m1!1e2!9m36!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e1!2b0!3e3!1m3!1e4!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e3'
        url_depthmap = URL_STR.format(panoid)

        r = requests.get(url_depthmap)
        resp = r.text[5:]
        json_data = json.loads(resp)

        size_img = (int(json_data[1][0][2][2][1]), int(json_data[1][0][2][2][0]))
        size_til = (0, 0)

        # # OLD ONE: As of Sep 2022, this one is not working anymore
        # r = requests.get(url_depthmap)  # getting the json file
        # json_data = r.json()
        # try:
        #     size_img = (int(json_data['Data']['image_width']), int(json_data['Data']['image_height']))
        #     size_til = (int(json_data['Data']['tile_width']), int(json_data['Data']['tile_height']))
        # except:
        #     print("The returned json could not be decoded")
        #     print(url_depthmap)
        #     print("status code: {}".format(r.status_code))
        #     return False, False, False

        return json_data, size_img, size_til


    def panoid_to_img(self, panoid, zoom, size_img, flip=False):
        GSV_TILEDIM = 512
        PANO_URL = 'https://maps.google.com/cbk?output=tile&panoid={panoid}&zoom={z}&x={x}&y={y}&' + str(datetime.now().microsecond)

        w, h = 2 ** zoom, 2 ** (zoom - 1)
        if size_img[0] == 13312: dim = 416
        if size_img[0] == 16384: dim = 512
        if not dim:
            print("!!!! THIS PANO IS A STRANGE DIMENSION {}".format(panoid))
            print("zoom:{}\t w,h: {}x{} \t image_size:{}x{}".format(zoom, w, h, size_img[0], size_img[1]))
            return False

        img = Image.new("RGB", (w * dim, h * dim), "red")
        try:
            for y in range(h):
                # if y % 5 == 0:
                #     print('{}/{}'.format(y, h))
                for x in range(w):
                    # print(y, x)
                    url_pano = PANO_URL.format(panoid=panoid, z=zoom, x=x, y=y)
                    response = requests.get(url_pano)
                    img_tile = Image.open(BytesIO(response.content))
                    img.paste(img_tile, (GSV_TILEDIM * x, GSV_TILEDIM * y))
        except:
            print("!!!! FAILED TO DOWNLOAD PANO for {}".format(panoid))
            return False

        if flip:
            print('FLIPPING LEFT-RIGHT TO MATCH DEPTH!')
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img
        

    def download_pano(self, panoid, zoom, output_filename, output_dir):
        depth_info, size_img, size_tile = self.panoid_to_depthinfo(panoid=panoid)
        pano_img = self.panoid_to_img(panoid, zoom=zoom, size_img=size_img, flip=False)

        if pano_img:
            pano_img.save(os.path.join(output_dir, output_filename))  # save pano
            return True
        else:
            print("!!!! FAILED\t{}".format(panoid))
            return False
        
        
        





