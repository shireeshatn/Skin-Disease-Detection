import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

# URL of the webpage
url = 'https://dermnetnz.org/images'

# Class name of the anchor tags
anchor_class = 'topicsList__group__items__item'  # Replace with the actual class name

async def scrape_js_enabled_anchors(url, anchor_class):
    async with async_playwright() as p:
        # Launch a browser
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Set a longer timeout (e.g., 60 seconds)
        await page.goto(url, timeout=120000)
        
        # Wait for JavaScript to execute and modify the DOM
        await page.wait_for_selector(f'.{anchor_class}', timeout=120000)
        
        # Get the updated page content
        content = await page.content()
        
        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find all anchor tags with the specified class
        anchor_tags = soup.find_all('a', class_=anchor_class)
        
        for anchor_tag in anchor_tags:
            href = anchor_tag.get('href')
            if href:
                print(href)
        
        # Close the browser
        await browser.close()

# Run the scraping function
asyncio.run(scrape_js_enabled_anchors(url, anchor_class))