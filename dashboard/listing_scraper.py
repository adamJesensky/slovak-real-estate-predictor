"""
Listing scraper for nehnutelnosti.sk and reality.sk.
Fetches a single listing URL, parses structured data (JSON-LD + HTML fallback),
and maps it to the model's input format for the dashboard.
"""

import re
import json
import unicodedata
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup


# --- Mapping dictionaries ---

# stav_final values must match exactly with model training data
VALID_STAV = {
    'Novostavba', 'Kompletná rekonštrukcia', 'Čiastočná rekonštrukcia',
    'Pôvodný stav', 'Vo výstavbe', 'Developerský projekt',
    'Určený k demolácii', 'undefined',
}

# Amenity keywords (Slovak) → model binary flags
AMENITY_KEYWORDS = {
    'has_lift': ['výťah'],
    'has_balcony': ['balkón', 'balkon'],
    'has_loggia': ['lodžia', 'loggia', 'logia'],
    'has_cellar': ['pivnica'],
    'has_garage': ['garáž', 'garaz'],
    'has_parking': ['parkovan', 'parkovacie'],
    'has_terrace': ['terasa'],
    'has_pantry': ['špajza', 'komora'],
    'has_warehouse': ['sklad'],
    'has_ac': ['klimatiz'],
}


def _strip_diacritics(text: str) -> str:
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.category(c).startswith('M'))


# Slovak room count words → number
_ROOM_WORDS = {
    'garson': 1, 'garzon': 1,
    'jednoizbov': 1, '1-izbov': 1,
    'dvojizbov': 2, '2-izbov': 2,
    'trojizbov': 3, '3-izbov': 3,
    'stvorizbov': 4, '4-izbov': 4,
    'patizbov': 5, '5-izbov': 5,
    'sestizbov': 6, '6-izbov': 6,
}


def _extract_room_count(text: str):
    """Extract room count from Slovak text (title, slug, meta description).
    E.g., 'dvojizbový byt' → 2, '3-izbový' → 3, 'garsónka' → 1.
    Returns int or None.
    """
    text_lower = _strip_diacritics(text.lower())
    for pattern, count in _ROOM_WORDS.items():
        if pattern in text_lower:
            return count
    # Generic pattern: N-izb or N izb
    m = re.search(r'(\d+)\s*-?\s*izb', text_lower)
    if m:
        return int(m.group(1))
    return None


# --- Fetching ---

def detect_portal(url: str):
    """Detect which portal the URL is from. Returns 'nehnutelnosti'/'reality' or None."""
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or '').lower()
        if parsed.scheme not in ('http', 'https'):
            return None
    except Exception:
        return None
    if hostname in ('nehnutelnosti.sk', 'www.nehnutelnosti.sk'):
        return 'nehnutelnosti'
    if hostname in ('reality.sk', 'www.reality.sk'):
        return 'reality'
    return None


def fetch_html(url: str):
    """Fetch listing HTML with proper headers. Returns (html, error_msg)."""
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/120.0.0.0 Safari/537.36'),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'sk-SK,sk;q=0.9,cs;q=0.8,en;q=0.7',
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        if resp.status_code == 404:
            return None, 'Inzerát nebol nájdený (404). Možno bol odstránený.'
        if resp.status_code == 403:
            return None, 'Prístup zamietnutý (403). Stránka blokuje automatické požiadavky.'
        resp.raise_for_status()
        return resp.text, None
    except requests.exceptions.Timeout:
        return None, 'Načítanie trvalo príliš dlho. Skúste to znova.'
    except requests.exceptions.ConnectionError:
        return None, 'Nepodarilo sa pripojiť k serveru. Skontrolujte internetové pripojenie.'
    except requests.exceptions.RequestException as e:
        return None, f'Chyba pri načítaní stránky: {str(e)[:100]}'


# --- Parsing ---

def _extract_jsonld(jd, data, _depth=0):
    """Recursively extract structured data from a JSON-LD object."""
    if _depth > 10 or not isinstance(jd, dict):
        return

    # Price from offers
    offers = jd.get('offers')
    if offers:
        offer = offers[0] if isinstance(offers, list) else offers
        if isinstance(offer, dict):
            price = offer.get('price')
            if price and 'price' not in data:
                try:
                    data['price'] = float(price)
                except (ValueError, TypeError):
                    pass

    # Floor size (direct property)
    fs = jd.get('floorSize')
    if isinstance(fs, dict) and 'value' in fs and 'floor_size' not in data:
        try:
            data['floor_size'] = float(fs['value'])
        except (ValueError, TypeError):
            pass

    # Amenity features — the main data source
    for feat in jd.get('amenityFeature', []):
        if not isinstance(feat, dict):
            continue
        name = feat.get('name', '')
        value = str(feat.get('value', ''))

        if name == 'Úžitková plocha':
            try:
                data['floor_size'] = float(re.sub(r'[^\d.,]', '', value).replace(',', '.'))
            except ValueError:
                pass
        elif name == 'Podlažie':
            try:
                data['current_floor'] = int(re.sub(r'[^\d]', '', value))
            except ValueError:
                pass
        elif name == 'Počet nadzemných podlaží':
            try:
                data['total_floors'] = int(re.sub(r'[^\d]', '', value))
            except ValueError:
                pass
        elif name in ('Počet izieb / miestností', 'Počet izieb'):
            try:
                data['room_count'] = int(re.sub(r'[^\d]', '', value))
            except ValueError:
                pass
        elif name == 'Stav nehnuteľnosti':
            data['stav'] = value
        elif name == 'Typ konštrukcie':
            data['construction_raw'] = value
        elif name == 'Vlastníctvo':
            data['vlastnictvo'] = value
        elif name == 'Vybavenie':
            data.setdefault('vybavenie_texts', [])
            data['vybavenie_texts'].append(value)
        elif name == 'Vykurovanie':
            data['heating_raw'] = value
        elif name == 'Plocha pozemku':
            try:
                data['land_area'] = float(re.sub(r'[^\d.,]', '', value).replace(',', '.'))
            except ValueError:
                pass
        elif name == 'Zastavaná plocha':
            try:
                data['built_up_area'] = float(re.sub(r'[^\d.,]', '', value).replace(',', '.'))
            except ValueError:
                pass

    # Recurse into nested structures
    for key in ('mainEntity', 'about'):
        nested = jd.get(key)
        if isinstance(nested, dict):
            _extract_jsonld(nested, data, _depth + 1)
        elif isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict):
                    _extract_jsonld(item, data, _depth + 1)

    for item in jd.get('itemListElement', []):
        if isinstance(item, dict):
            inner = item.get('mainEntity') or item.get('item') or item
            if isinstance(inner, dict):
                _extract_jsonld(inner, data, _depth + 1)


def _extract_json_value(text: str, start: int) -> str:
    """Extract a balanced JSON value (object or array) starting at `start`.
    Returns the substring or empty string if no balanced structure found."""
    if start >= len(text):
        return ''
    opener = text[start]
    if opener == '{':
        closer = '}'
    elif opener == '[':
        closer = ']'
    else:
        return ''
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ''


def _extract_rsc_payload(html: str, data: dict):
    """Extract listing data from Next.js RSC payload (nehnutelnosti.sk).

    nehnutelnosti.sk uses React Server Components. The full advertisement data
    is embedded in self.__next_f.push([1, "..."]) script chunks. The payload
    uses RSC wire format — a mix of component references and inline JSON props.
    We find the "parameters" object within these chunks and extract key fields.
    """
    # Find all self.__next_f.push chunks
    # The regex must handle JS-escaped quotes (\") inside the string literal.
    # (?:[^"\\]|\\.)* matches non-quote/non-backslash chars OR escaped chars.
    chunks = re.findall(r'self\.__next_f\.push\(\[1,\s*"((?:[^"\\]|\\.)*)"\]', html, re.DOTALL)
    if not chunks:
        return

    # Unescape JS string escapes
    def _unescape_js(s):
        s = s.replace('\\"', '"')
        s = s.replace('\\\\', '\\')
        s = s.replace('\\n', '\n')
        s = s.replace('\\t', '\t')
        return s

    for raw_chunk in chunks:
        chunk = _unescape_js(raw_chunk)

        # --- Extract "parameters" object ---
        # In the RSC payload, the ad data appears as: ..."parameters":{...}...
        # We find this key and extract the balanced JSON object after it.
        params_match = re.search(r'"parameters"\s*:\s*\{', chunk)
        if not params_match:
            continue

        # Check this is the advertisement parameters (has relevant fields nearby)
        context_after = chunk[params_match.start():params_match.start() + 2000]
        if not any(kw in context_after for kw in (
            '"construction"', '"area"', '"realEstateState"',
            '"floor"', '"transaction"', '"hasElevator"',
        )):
            continue

        # Extract the parameters JSON object
        obj_start = params_match.end() - 1  # point at the opening {
        params_str = _extract_json_value(chunk, obj_start)
        if not params_str:
            continue

        try:
            params = json.loads(params_str, strict=False)
        except json.JSONDecodeError:
            continue

        if not isinstance(params, dict):
            continue

        # --- Extract from parameters ---
        if 'construction_raw' not in data:
            val = params.get('construction')
            if val:
                data['construction_raw'] = str(val)

        if 'current_floor' not in data:
            fl = params.get('floor')
            if fl is not None:
                try:
                    data['current_floor'] = int(fl)
                except (ValueError, TypeError):
                    pass

        if 'total_floors' not in data:
            nf = params.get('numberOfFloors')
            if nf is not None:
                try:
                    data['total_floors'] = int(nf)
                except (ValueError, TypeError):
                    pass

        if 'floor_size' not in data:
            area = params.get('area')
            if area is not None:
                try:
                    data['floor_size'] = float(str(area).replace(',', '.'))
                except (ValueError, TypeError):
                    pass

        if 'stav' not in data:
            stav = params.get('realEstateState')
            if stav:
                data['stav'] = str(stav)

        if 'year_of_construction' not in data:
            yoc = params.get('yearOfConstruction')
            if yoc is not None:
                try:
                    data['year_of_construction'] = int(yoc)
                except (ValueError, TypeError):
                    pass

        # Price — nested under parameters.price in RSC format
        if 'price' not in data:
            price_obj = params.get('price')
            if isinstance(price_obj, dict):
                pn = price_obj.get('priceNum')
                if pn is not None:
                    try:
                        data['price'] = float(pn)
                    except (ValueError, TypeError):
                        pass
                if 'price' not in data:
                    pv = price_obj.get('priceValue')
                    if pv is not None:
                        try:
                            data['price'] = float(str(pv).replace(' ', ''))
                        except (ValueError, TypeError):
                            pass

        # Attributes array: heating, vlastníctvo, room count, etc.
        attrs = params.get('attributes')
        if isinstance(attrs, list):
            for attr in attrs:
                if not isinstance(attr, dict):
                    continue
                label = str(attr.get('label', ''))
                value = str(attr.get('value', ''))
                if not label or not value:
                    continue

                if 'Vlastníctvo' in label and 'vlastnictvo' not in data:
                    data['vlastnictvo'] = value
                elif 'ykurovanie' in label and 'heating_raw' not in data:
                    data['heating_raw'] = value
                elif 'izieb' in label.lower() and 'room_count' not in data:
                    try:
                        data['room_count'] = int(re.sub(r'[^\d]', '', value))
                    except ValueError:
                        pass
                elif 'pivn' in label.lower() and 'has_cellar_rsc' not in data:
                    data['has_cellar_rsc'] = True
                elif 'Vybavenie' in label or 'ybavenie' in label:
                    data.setdefault('vybavenie_texts', [])
                    data['vybavenie_texts'].append(value)

        # hasElevator boolean — add after attributes to avoid duplicates
        if params.get('hasElevator') is True:
            data.setdefault('vybavenie_texts', [])
            if not any('výťah' in t.lower() for t in data['vybavenie_texts']):
                data['vybavenie_texts'].append('Výťah')

        # --- Extract location / GPS from the same chunk ---
        # The property location object has "city", "district", "point" fields.
        # We find the location block that contains "city" (to avoid matching
        # the advertiser's location which only has "parts").
        for loc_match in re.finditer(r'"location"\s*:\s*\{', chunk):
            loc_start = loc_match.end() - 1
            loc_str = _extract_json_value(chunk, loc_start)
            if not loc_str or '"city"' not in loc_str:
                continue
            try:
                loc_obj = json.loads(loc_str, strict=False)
            except json.JSONDecodeError:
                continue
            if not isinstance(loc_obj, dict) or 'city' not in loc_obj:
                continue

            # GPS coordinates
            point = loc_obj.get('point')
            if isinstance(point, dict):
                if 'lat' not in data and 'latitude' in point:
                    data['lat'] = str(point['latitude'])
                if 'lon' not in data and 'longitude' in point:
                    data['lon'] = str(point['longitude'])

            # City name (most useful for matching)
            if 'location_raw' not in data:
                city = loc_obj.get('city')
                if city:
                    data['location_raw'] = str(city)
            break

        # Category from the "category" field near the advertisement
        if 'category_raw' not in data:
            cat_match = re.search(r'"category"\s*:\s*\{[^}]*"name"\s*:\s*"([^"]+)"', chunk)
            if cat_match:
                cat_name = cat_match.group(1)
                if 'byt' in cat_name.lower():
                    data['category_raw'] = 'byty'
                elif 'dom' in cat_name.lower() or 'chat' in cat_name.lower():
                    data['category_raw'] = 'domy'

        # Found the main parameters — done with RSC extraction
        break


def parse_detail_html(html: str) -> dict:
    """Parse listing detail page HTML. Returns dict with extracted values."""
    soup = BeautifulSoup(html, 'html.parser')
    data = {}

    # 0. Next.js RSC payload — nehnutelnosti.sk embeds full data in self.__next_f.push chunks
    _extract_rsc_payload(html, data)

    # 1. JSON-LD — primary structured data source (works on reality.sk, empty on nehnutelnosti.sk)
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            if not script.string:
                continue
            jd = json.loads(script.string, strict=False)
            _extract_jsonld(jd, data)
        except (json.JSONDecodeError, TypeError):
            continue

    # 2. page-info div — reliable category/location metadata
    page_info = soup.find('div', id='page-info')
    if page_info:
        cat1 = page_info.get('data-offer-cat1')
        if cat1:
            data['category_raw'] = cat1
        cat3 = page_info.get('data-offer-cat3')
        if cat3:
            data['location_raw'] = cat3
        cat4 = page_info.get('data-offer-cat4')
        if cat4:
            data['transaction'] = cat4
        cat5 = page_info.get('data-offer-cat5')
        if cat5:
            data.setdefault('stav', cat5)

    # 3. GPS coordinates from map div
    map_div = soup.find('div', id='js-map-detail')
    if map_div:
        lat = map_div.get('data-latitude')
        lon = map_div.get('data-longitude')
        if lat:
            data['lat'] = lat
        if lon:
            data['lon'] = lon

    # 4. HTML fallback — info-title/info-value pairs
    for title_div in soup.find_all('div', class_='info-title'):
        key = title_div.get_text(strip=True).rstrip(':')
        if not key:
            continue
        val_div = title_div.find_next_sibling('div')
        if val_div:
            val = val_div.get_text(separator=' ', strip=True)
            # Only fill what we don't have yet from JSON-LD
            if key == 'Úžitková plocha' and 'floor_size' not in data:
                try:
                    data['floor_size'] = float(re.sub(r'[^\d.,]', '', val).replace(',', '.'))
                except ValueError:
                    pass

    # 5. Title for display
    title_tag = soup.find('title')
    if title_tag:
        data['title'] = title_tag.get_text(strip=True)

    # 6. Meta tags — fallback for client-side rendered pages (nehnutelnosti.sk React SPA)
    for meta in soup.find_all('meta'):
        name = meta.get('name', '').lower()
        prop = meta.get('property', '').lower()
        content = meta.get('content', '')
        if not content:
            continue

        if (name == 'description' or prop == 'og:description') and 'meta_description' not in data:
            data['meta_description'] = content
            _parse_meta_description(content, data)

        if prop == 'og:title' and 'title' not in data:
            data['title'] = content

    # 7. Breadcrumb links for category (nehnutelnosti.sk uses /vysledky/byty paths)
    if 'category_raw' not in data:
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].lower()
            if '/vysledky/' in href or '/predaj/' in href:
                if '/byty' in href:
                    data['category_raw'] = 'byty'
                    break
                if '/domy' in href:
                    data['category_raw'] = 'domy'
                    break

    # 8. Room count from title / meta description (fallback for SPA pages)
    if 'room_count' not in data:
        for text_src in [data.get('title', ''), data.get('meta_description', '')]:
            rc = _extract_room_count(text_src)
            if rc:
                data['room_count'] = rc
                break

    return data


def _parse_meta_description(content: str, data: dict):
    """Extract structured info from meta description.
    Common format: '2 izbový byt, Predaj, Bratislava-Nové Mesto, Novostavba, 60 m², 376 341 €'
    """
    parts = [p.strip() for p in content.split(',')]
    for part in parts:
        # Price: "249 000 €" or "376 341 €" (may use non-breaking space \xa0)
        price_match = re.search(r'([\d\s\xa0]+)\s*€', part)
        if price_match and 'price' not in data:
            try:
                price_str = re.sub(r'\s+', '', price_match.group(1))
                data['price'] = float(price_str)
            except ValueError:
                pass
        # Area: "72.3 m²" or "60 m²"
        area_match = re.search(r'([\d.,]+)\s*m[²2]', part)
        if area_match and 'floor_size' not in data:
            try:
                data['floor_size'] = float(area_match.group(1).replace(',', '.'))
            except ValueError:
                pass
        # Stav: check if part matches known values (e.g., "Novostavba", "Pôvodný stav")
        part_clean = part.strip()
        if part_clean in VALID_STAV and 'stav' not in data:
            data['stav'] = part_clean
    # Location — typically the 3rd comma-separated part (after type and transaction)
    if len(parts) >= 3 and 'location_raw' not in data:
        loc_part = parts[2].strip()
        if loc_part and not re.search(r'[€m²]|\d{3,}', loc_part):
            data['location_raw'] = loc_part


# --- Category detection ---

def detect_category(parsed: dict, url: str):
    """Detect if listing is byty or domy. Returns 'byty'/'domy' or None."""
    url_lower = url.lower()

    # 1. URL path segments (e.g., /byty/predaj/, /domy/...)
    if '/byty/' in url_lower or '/byt/' in url_lower:
        return 'byty'
    if '/domy/' in url_lower or '/dom/' in url_lower or '/chaty/' in url_lower:
        return 'domy'

    # 2. page-info / breadcrumb category
    cat = parsed.get('category_raw', '').lower()
    if 'byt' in cat:
        return 'byty'
    if 'dom' in cat or 'chat' in cat or 'chalup' in cat:
        return 'domy'

    # 3. URL slug keywords (for /detail/{id}/{slug} format on nehnutelnosti.sk)
    slug = url_lower.rstrip('/').split('/')[-1]
    slug_words = set(slug.split('-'))
    if 'byt' in slug_words:
        return 'byty'
    if 'dom' in slug_words or 'chata' in slug_words or 'chalupa' in slug_words:
        return 'domy'

    # 4. Meta description (first comma part often has "2 izbový byt" etc.)
    meta = parsed.get('meta_description', '').lower()
    if meta:
        first_part = meta.split(',')[0]
        if 'byt' in first_part:
            return 'byty'
        if 'dom' in first_part or 'chat' in first_part:
            return 'domy'

    return None


# --- Feature mapping ---

def map_construction(raw: str) -> str:
    """Map scraped construction text to model's English label.
    Logic mirrors parse_construction_type() from feature_engineering_v3.py."""
    if not raw:
        return 'Unknown'
    x = raw.lower()
    if 'tehl' in x:
        return 'Brick'
    if 'panel' in x:
        return 'Panel'
    if 'zmiešan' in x:
        return 'Mixed'
    if 'železobet' in x:
        return 'ReinforcedConcrete'
    if 'dreven' in x:
        return 'Wood'
    if 'skelet' in x:
        return 'Skeleton'
    if 'kváder' in x or 'blok' in x:
        return 'Block'
    if 'kamenn' in x:
        return 'Stone'
    if 'montovan' in x:
        return 'Prefab'
    if 'hlin' in x:
        return 'Block'
    return 'Other'


def map_heating(raw: str) -> str:
    """Map scraped heating text to model's label.
    Logic mirrors get_heating_type() from feature_engineering_v3.py."""
    if not raw:
        return 'unknown'
    x = _strip_diacritics(raw).lower()
    if 'podlahov' in x:
        return 'underfloor'
    if 'ustredne' in x or 'centralne' in x:
        return 'central'
    if 'lokalne' in x:
        return 'local'
    return 'other'


def parse_amenities(vybavenie_texts: list) -> dict:
    """Parse vybavenie text list into binary amenity flags."""
    combined = ' '.join(vybavenie_texts).lower()
    result = {}
    for key, keywords in AMENITY_KEYWORDS.items():
        result[key] = any(kw in combined for kw in keywords)
    return result


# --- Location matching ---

def match_location(location_str: str, locations_dict: dict):
    """
    Match scraped location string to our mappings.
    Returns (matched_name, confidence) where confidence is 'high'/'medium'/'low'.
    """
    if not location_str:
        return None, 'low'

    # Parse: "Slovensko, Bratislavský kraj, okres Bratislava V, Bratislava-Petržalka"
    parts = [p.strip() for p in location_str.split(',')]
    all_locations = list(locations_dict.keys())

    # Try from most specific (last) to least specific
    candidates = list(reversed(parts))

    for candidate in candidates:
        if candidate.lower() in ('slovensko', ''):
            continue
        # Strip "okres " prefix
        clean = re.sub(r'^okres\s+', '', candidate).strip()

        # 1. Exact match
        if clean in locations_dict:
            return clean, 'high'

        # 2. Case-insensitive
        for loc in all_locations:
            if loc.lower() == clean.lower():
                return loc, 'high'

        # 3. Diacritics-insensitive
        clean_nd = _strip_diacritics(clean).lower()
        for loc in all_locations:
            if _strip_diacritics(loc).lower() == clean_nd:
                return loc, 'high'

        # 4. Contains match — collect all candidates, pick shortest (most specific)
        _contains_hits = []
        for loc in all_locations:
            loc_nd = _strip_diacritics(loc).lower()
            if clean_nd == loc_nd or (len(clean_nd) > 3 and clean_nd in loc_nd):
                _contains_hits.append(loc)
            elif len(loc_nd) > 3 and loc_nd in clean_nd:
                _contains_hits.append(loc)
        if _contains_hits:
            best = min(_contains_hits, key=len)
            return best, 'medium'

    # 5. Last resort: match just the city name (first word of most specific part)
    if parts:
        city_part = parts[-1].strip()
        city_base = city_part.split('-')[0].split(' ')[0]
        if len(city_base) > 2:
            city_nd = _strip_diacritics(city_base).lower()
            matches = [loc for loc in all_locations
                       if _strip_diacritics(loc).lower().startswith(city_nd)]
            if len(matches) == 1:
                return matches[0], 'low'
            elif matches:
                return matches[0], 'low'

    return None, 'low'


# --- Main entry point ---

def fetch_listing(url: str) -> dict:
    """
    Main entry point. Fetches and parses a listing URL.

    Returns dict with:
        - success: bool
        - error: str (if failed)
        - category: 'byty'/'domy'
        - parsed: dict of raw parsed values
        - title: str
    """
    # Validate portal
    portal = detect_portal(url)
    if not portal:
        return {'success': False,
                'error': 'Nepodporovaný portál. Použite odkaz z nehnutelnosti.sk alebo reality.sk.'}

    # Fetch HTML
    html, err = fetch_html(url)
    if err:
        return {'success': False, 'error': err}

    # Detect removed listings (nehnutelnosti.sk Next.js redirect)
    if 'NEXT_REDIRECT' in html and '/vysledky' in html:
        return {'success': False,
                'error': 'Inzerát bol odstránený alebo nie je dostupný.'}

    # Parse
    parsed = parse_detail_html(html)
    if not parsed:
        return {'success': False,
                'error': 'Nepodarilo sa extrahovať údaje zo stránky.'}

    # reality.sk may have different HTML structure — warn if little data extracted
    _data_keys = {k for k in parsed if k != 'title'}
    if portal == 'reality' and len(_data_keys) < 2:
        return {'success': False,
                'error': 'Podpora pre reality.sk je experimentálna. '
                         'Nepodarilo sa extrahovať údaje. Skúste odkaz z nehnutelnosti.sk.'}

    # Detect category
    category = detect_category(parsed, url)
    if not category:
        return {'success': False,
                'error': 'Nepodporovaná kategória. Podporujeme len byty a domy.'}

    # Check transaction type
    transaction = parsed.get('transaction', '').lower()
    if transaction and transaction != 'predaj':
        return {'success': False,
                'error': f'Inzerát je typu "{parsed["transaction"]}". Podporujeme len predaj.'}

    # Check we have at least floor_size with a sane value
    floor_size = parsed.get('floor_size', 0)
    if not floor_size or floor_size < 8:
        return {'success': False,
                'error': 'Nepodarilo sa nájsť plochu nehnuteľnosti v inzeráte.'}

    # Try to extract room count from URL slug as last resort
    if 'room_count' not in parsed:
        slug = url.rstrip('/').split('/')[-1]
        rc = _extract_room_count(slug.replace('-', ' '))
        if rc:
            parsed['room_count'] = rc

    return {
        'success': True,
        'category': category,
        'parsed': parsed,
        'title': parsed.get('title', 'Inzerát'),
    }


def build_model_input(parsed: dict, category: str, mappings: dict) -> tuple:
    """
    Convert parsed listing data to dashboard input format.

    Returns (input_dict, warnings, loc_confidence).
    - input_dict: ready for session_state / process_input()
    - warnings: list of Slovak warning messages
    - loc_confidence: 'high'/'medium'/'low'
    """
    warnings = []

    # --- Location ---
    location_str = parsed.get('location_raw', '')
    matched_loc, loc_confidence = match_location(location_str, mappings.get('locations', {}))

    if not matched_loc:
        all_locs = sorted(mappings.get('locations', {}).keys())
        matched_loc = all_locs[0] if all_locs else ''
        warnings.append(f'Lokalitu "{location_str.split(",")[-1].strip()}" sa nepodarilo priradiť. '
                        'Vyberte manuálne.')
    elif loc_confidence == 'low':
        warnings.append(f'Lokalita "{matched_loc}" je len približná zhoda. Odporúčame skontrolovať.')

    # --- Stav ---
    stav = parsed.get('stav', '')
    if stav not in VALID_STAV:
        if stav:
            warnings.append(f'Stav "{stav}" nebol rozpoznaný, použitý "Pôvodný stav".')
        else:
            warnings.append('Stav nehnuteľnosti nebol v inzeráte uvedený.')
        stav = 'Pôvodný stav'

    # --- Construction ---
    construction = map_construction(parsed.get('construction_raw', ''))
    if construction in ('Unknown', 'Other') and not parsed.get('construction_raw'):
        warnings.append('Typ konštrukcie nebol v inzeráte uvedený.')

    # --- Floor size ---
    floor_size = parsed.get('floor_size', 0)
    if not floor_size or floor_size <= 0:
        warnings.append('Plocha nebola nájdená v inzeráte.')
        floor_size = 60 if category == 'byty' else 150

    # --- Room count ---
    room_count = parsed.get('room_count', 0)
    if not room_count:
        warnings.append('Počet izieb nebol uvedený.')
        room_count = 2

    # --- Floors ---
    current_floor = parsed.get('current_floor', 0)
    total_floors = parsed.get('total_floors', 0)
    if category == 'byty':
        if not total_floors:
            warnings.append('Počet poschodí nebol uvedený.')
            total_floors = 5

    # --- Heating ---
    heating = map_heating(parsed.get('heating_raw', ''))
    if heating == 'unknown':
        warnings.append('Vykurovanie nebolo v inzeráte uvedené.')

    # --- Vlastnictvo ---
    vlastnictvo = parsed.get('vlastnictvo', 'Osobné')
    if vlastnictvo not in ('Osobné', 'Firemné'):
        vlastnictvo = 'Osobné'

    # --- Amenities ---
    amenities = parse_amenities(parsed.get('vybavenie_texts', []))

    # --- Land area (domy only) ---
    land_area = int(parsed.get('land_area', 0)) if category == 'domy' else 0
    built_up_area = int(parsed.get('built_up_area', 0)) if category == 'domy' else 0

    # --- Price ---
    price = int(parsed.get('price', 0))

    input_dict = {
        'floor_size': int(floor_size),
        'obec_cast': matched_loc,
        'stav_final': stav,
        'construction': construction,
        'room_count': int(room_count),
        'current_floor': int(current_floor),
        'total_floors': int(total_floors),
        'has_lift': int(amenities.get('has_lift', False)),
        'has_balcony': int(amenities.get('has_balcony', False)),
        'has_loggia': int(amenities.get('has_loggia', False)),
        'has_cellar': int(amenities.get('has_cellar', False) or parsed.get('has_cellar_rsc', False)),
        'has_garage': int(amenities.get('has_garage', False)),
        'has_parking': int(amenities.get('has_parking', False)),
        'has_terrace': int(amenities.get('has_terrace', False)),
        'has_pantry': int(amenities.get('has_pantry', False)),
        'has_warehouse': int(amenities.get('has_warehouse', False)),
        'has_ac': int(amenities.get('has_ac', False)),
        'land_area': land_area,
        'built_up_area': built_up_area,
        'has_gas': 1,
        'has_water': 1,
        'has_electricity': 1,
        'has_sewerage': 1,
        'heating': heating,
        'vlastnictvo': vlastnictvo,
        'market_price': price,
    }

    return input_dict, warnings, loc_confidence
