# -*- coding: utf-8 -*-
import os
import click
import logging
import subprocess
import zipfile
import tarfile
import shutil
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Extract the zip files from raw folder to processed
    if not os.listdir(output_filepath) or (os.listdir(output_filepath) == ['.gitkeep']):
        for file_name in os.listdir(input_filepath):
            if file_name.endswith('.7z'):
                subprocess.call(['7z', 'x', '-r', '-y', '-o%s' % output_filepath,
                                 os.path.join(input_filepath, file_name)], shell=False)
            if file_name.endswith('.zip'):
                with zipfile.ZipFile(os.path.join(input_filepath, file_name)) as zf:
                    zf.extractall(output_filepath)

    # Extract the tar files in processed folder
    for file_name in os.listdir(output_filepath):
        if file_name.endswith('.tar'):
            tar = tarfile.open(os.path.join(output_filepath, file_name))
            tar.extractall(output_filepath)
            tar.close()
            os.remove(os.path.join(output_filepath, file_name))

    # Delete MAC_OS folder
    if os.path.isdir(os.path.join(output_filepath, '__MACOSX')):
        shutil.rmtree(os.path.join(output_filepath, '__MACOSX'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
