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


def parse_detail_html(html: str) -> dict:
    """Parse listing detail page HTML. Returns dict with extracted values."""
    soup = BeautifulSoup(html, 'html.parser')
    data = {}

    # 1. JSON-LD — primary structured data source
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

    return data


# --- Category detection ---

def detect_category(parsed: dict, url: str):
    """Detect if listing is byty or domy. Returns 'byty'/'domy' or None."""
    url_lower = url.lower()
    if '/byty/' in url_lower or '/byt/' in url_lower:
        return 'byty'
    if '/domy/' in url_lower or '/dom/' in url_lower or '/chaty/' in url_lower:
        return 'domy'

    cat = parsed.get('category_raw', '').lower()
    if 'byt' in cat:
        return 'byty'
    if 'dom' in cat or 'chat' in cat or 'chalup' in cat:
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
        'has_cellar': int(amenities.get('has_cellar', False)),
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
