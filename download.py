import tarfile
from landsatxplore.earthexplorer import EarthExplorerError
from pathlib import Path
import tarfile
import yaml
from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer

config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)


def download_scenes(api, earth_explorer, latitude, longitude, start_date, end_date, max_cloud_cover, output_dir):
    scenes = api.search(
        dataset='landsat_tm_c2_l2',
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=max_cloud_cover
    )

    print(f"{len(scenes)} scenes found.")

    output_dir = Path(output_dir) 
    output_dir.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        scene_file = output_dir / f"{scene['display_id']}.tar" 

        if scene_file.exists():
            print(f"Scene {scene['display_id']} already downloaded.")
            continue

        else:
            print(f"Downloading scene {scene['display_id']}...")
            try:
                earth_explorer.download(scene['entity_id'], output_dir=output_dir, dataset='landsat_tm_c2_l2', timeout=1000)
            except EarthExplorerError as e:
                print(e)

    return scenes


def untar_scenes(output_dir):
    output_dir = Path(output_dir)  

    for tar_file in output_dir.glob("*.tar"):  
        print(f"Extracting {tar_file.name}...")
        
        with tarfile.open(tar_file) as tar:
            tar.extractall(path=output_dir)

        # Uncomment if you would like to delete the tar files
        # print(f"Deleting {tar_file.name}...")
        # tar_file.unlink()


def main():
    api            = API(config['usgs']['username'], config['usgs']['password'])
    earth_explorer = EarthExplorer(config['usgs']['username'], config['usgs']['password'])

    # Define the region (Tempe, Arizona)
    latitude, longitude = config['coordinates']['latitude'], config['coordinates']['longitude']

    # Define the date interval
    start_date = str(config['time_period']['start_date'])
    end_date   = str(config['time_period']['end_date'])

    # Output path for the downloaded scenes
    output_dir = './temp/landsat_scenes/'

    download_scenes(api, earth_explorer, latitude, longitude, start_date, end_date, 10, output_dir)
    # untar_scenes(output_dir)

    api.logout()


if __name__ == "__main__":
    main()
