import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote

def web_search(query, max_results=5):
    results = []

    # Google News RSS
    try:
        gurl = "https://news.google.com/rss/search?q=" + quote(query) + "&hl=vi&gl=VN&ceid=VN:vi"
        r = requests.get(gurl, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        root = ET.fromstring(r.content)
        for item in root.findall(".//item")[:max_results]:
            title   = item.findtext("title", "").strip()
            link    = item.findtext("link", "").strip()
            pubdate = item.findtext("pubDate", "").strip()
            if title:
                results.append(f"📰 {title}\n  🕐 {pubdate}\n  🔗 {link}")
    except requests.exceptions.RequestException as e:
        results.append("Loi Google News: " + str(e))
    except Exception as e:
        results.append("Loi Google News: " + str(e))

    # Bing News RSS (bo sung neu Google it ket qua)
    if len(results) < max_results:
        try:
            burl = "https://www.bing.com/news/search?q=" + quote(query) + "&format=RSS"
            r = requests.get(burl, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:max_results - len(results)]:
                title   = item.findtext("title", "").strip()
                link    = item.findtext("link", "").strip()
                pubdate = item.findtext("pubDate", "").strip()
                if title:
                    results.append(f"📰 {title}\n  🕐 {pubdate}\n  🔗 {link}")
        except requests.exceptions.RequestException as e:
            results.append("Loi Bing News: " + str(e))
        except Exception as e:
            results.append("Loi Bing News: " + str(e))

    if not results:
        return "Khong tim thay ket qua cho: " + query
    return "\n\n".join(results)

print(web_search("Tin tức Hôm nay"))