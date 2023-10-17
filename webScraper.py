import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def scrape_website(url, visited=set()):
    print("iteration")
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will throw an HTTPError if the response was unsuccessful
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err} - URL: {url}")
        return  # Exit the function, skipping the rest of the code

    soup = BeautifulSoup(response.content, 'html.parser')
    visited.add(url)

    # Save content to file
    text_content = '\n'.join(element.text for element in soup.find_all(['p']))
    with open('dailyIllini2.txt', 'a', encoding='utf-8') as file:
        file.write(f'\n\n--- Content from {url} ---\n\n{text_content}\n')

    # Find all links on the page
    for a_tag in soup.find_all('a', href=True):
        link = urljoin(url, a_tag['href'])
        # Only visit links within the same domain and not visited yet
        if url in link and link not in visited:
            scrape_website(link, visited)

# Usage:
url = 'https://dailyillini.com/'  # Replace with the URL of the website you want to scrape
scrape_website(url)
