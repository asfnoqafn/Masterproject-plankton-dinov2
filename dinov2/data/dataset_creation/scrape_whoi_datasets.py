import argparse
import logging
import os

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    args = parser.parse_args()
    return args.url, args.output_path


def set_driver(url):
    options = FirefoxOptions()
    options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)
    driver.get(url)
    return driver


def page_to_soup(driver):
    # As string:
    html_content = driver.page_source
    return BeautifulSoup(html_content, 'html.parser')


def download_zip(driver, soup, output_path):
    bin_element = soup.find(id="bin-header")
    href = bin_element.get('href') if bin_element else None
    bin_name = href.split('=')[-1] if href else 'default'
    zip_link = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "download-zip"))).get_attribute('href')
    response = requests.get(zip_link)
    with open(os.path.join(output_path, f'{bin_name}.zip'), 'ab') as file:
        file.write(response.content)
    logging.info(f'Saved zip: {bin_name}')
    return bin_name


def click_next(driver, bin_name):
    try:
        button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "next-bin")))
        driver.execute_script("arguments[0].scrollIntoView(true);", button)
        driver.execute_script("arguments[0].click();", button)
        return True
    except Exception as e:
        logging.error(f'Failed to click next for bin: {bin_name} with error {e}')
        return False


def main():
    url, output_path = get_args()
    os.makedirs(output_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(output_path, 'bin_log.txt'),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    driver = set_driver(url)
    while True:
        soup = page_to_soup(driver)
        bin_name = download_zip(driver, soup, output_path)
        if not click_next(driver, bin_name):
            break

    driver.quit()


main()
