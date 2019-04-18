from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
books = []
def get_moby_dick():
        text = strip_headers(load_etext(2071)).strip()
        return text

def get_texts(n):
    books = []
    i = 0
    j = 0
    while i < n:
        try:
            text = strip_headers(load_etext(j)).strip()
            books.append(text)
            i += 1
        except:
            pass
        j += 1
    return books


if __name__ == "__main__":
    for i in range(0,10000000):
        try:
            text = strip_headers(load_etext(i)).strip()
            print(text)
            print(i)
            books.append(i)
        except:
            pass
