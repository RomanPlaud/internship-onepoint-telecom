import requests
from bs4 import BeautifulSoup

response = requests.get(
	url="https://en.wikipedia.org/wiki/Category:Anguilla_National_Alliance_politicians",
)

## Find all the categories
soup = BeautifulSoup(response.content, 'html.parser')

# ## find in the page the category all the hyperlinks
categories = soup.find_all('div', class_='mw-category-group')

## extract the hyperlinks
for category in categories:
    links = category.find_all('a')
    for link in links:
        print(link.get('href'))
    print('------------------')
    #     ## access the link and extract the introduction
    #     response = requests.get(url="https://en.wikipedia.org" + link.get('href'))
    #     soup = BeautifulSoup(response.content, 'html.parser')
    #     # get the first paragraph
    #     print(soup.find('p').text)
        
