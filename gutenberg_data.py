from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
books = []
for i in range(0,10000000):
    try:
        text = strip_headers(load_etext(i)).strip()
        print(text)
        print(i)
        books.append(i)
    except:
        pass

def get_moby_dick():
        text = strip_headers(load_etext(2071)).strip()
        return text
