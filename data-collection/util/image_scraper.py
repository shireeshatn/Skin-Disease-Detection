import asyncio
import os
from urllib.parse import urljoin
from pathlib import Path
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import requests

base_url = 'https://dermnetnz.org'

# Read URLs from the text file
with open('data-collection/disease_images_links.txt', 'r') as file:
    relative_urls = [line.strip() for line in file]

async def download_image(image_url, folder):
    """Download an image from a URL and save it to a folder."""
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image_name = os.path.join(folder, image_url.split('/')[-1])
            with open(image_name, 'wb') as file:
                file.write(response.content)
    except Exception as e:
        print(f'Error downloading {image_url}: {e}')

async def scrape_images_from_page(url, disease_name):
    folder_path = Path('data-collection/dataset/' + disease_name)
    folder_path.mkdir(parents=True, exist_ok=True)
    
    async with async_playwright() as p:
        response = requests.get(url)

        if response.status_code == 200:
            # Parse the page content with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
        
            # Find all image tags
            image_tags = soup.find_all('img')
        
            # Extract and download images
            for img_tag in image_tags:
                img_src = img_tag.get('src')
                if img_src:
                    img_url = urljoin(base_url, img_src)
                    await download_image(img_url, folder_path)

# Main function to iterate over all URLs and scrape images
async def main():
    tasks = []
    for relative_url in relative_urls:
        disease_name = relative_url.split('/')[-1].replace('-images', '')
        full_url = urljoin(base_url, relative_url)
        tasks.append(scrape_images_from_page(full_url, disease_name))
    
    await asyncio.gather(*tasks)

# Run the main function
asyncio.run(main())
