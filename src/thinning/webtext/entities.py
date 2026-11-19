"""Names, prompts and answer groups of the two factual relations of web text.

Capitals (``cap``): the answer to a country is its capital. City countries (``ctry``): the answer to a city that
is not a capital is its country. Each entry lists the canonical name first, which the evaluation prompts use,
followed by the aliases that the corpus labeller also matches. A leading "the" belongs to the prompt form only.

The evaluation prompts fill 8 templates with every kept subject, and an answer is the group of the first tokens of
its names. Countries named with "the" are not answers of city prompts, and a country is kept with two of its
cities. Answers whose group overlaps another group, subjects that share their first four letters with the answer
and prompts that contain a token of the answer group are left out, which leaves 816 capital prompts over 102
answer groups and 864 city prompts over 54 groups.
"""

from collections import Counter

# (country names, capital names)
CAPITALS = [
    (["Afghanistan"], ["Kabul"]), (["Albania"], ["Tirana"]), (["Angola"], ["Luanda"]),
    (["Argentina"], ["Buenos Aires"]), (["Armenia"], ["Yerevan"]), (["Australia"], ["Canberra"]),
    (["Austria"], ["Vienna"]), (["Azerbaijan"], ["Baku"]), (["Bahrain"], ["Manama"]),
    (["Bangladesh"], ["Dhaka", "Dacca"]), (["Belarus"], ["Minsk"]), (["Belgium"], ["Brussels"]),
    (["Bosnia and Herzegovina", "Bosnia"], ["Sarajevo"]), (["Botswana"], ["Gaborone"]), (["Bulgaria"], ["Sofia"]),
    (["Cambodia"], ["Phnom Penh"]), (["Cameroon"], ["Yaoundé", "Yaounde"]), (["Canada"], ["Ottawa"]),
    (["Chile"], ["Santiago"]), (["China"], ["Beijing", "Peking"]), (["Colombia"], ["Bogotá", "Bogota"]),
    (["Croatia"], ["Zagreb"]), (["Cuba"], ["Havana"]), (["Cyprus"], ["Nicosia"]),
    (["the Czech Republic", "Czech Republic", "Czechia"], ["Prague"]), (["Denmark"], ["Copenhagen"]),
    (["Ecuador"], ["Quito"]), (["Egypt"], ["Cairo"]), (["Estonia"], ["Tallinn"]), (["Ethiopia"], ["Addis Ababa"]),
    (["Fiji"], ["Suva"]), (["Finland"], ["Helsinki"]), (["France"], ["Paris"]), (["Germany"], ["Berlin"]),
    (["Ghana"], ["Accra"]), (["Greece"], ["Athens"]), (["Honduras"], ["Tegucigalpa"]), (["Hungary"], ["Budapest"]),
    (["Iceland"], ["Reykjavik", "Reykjavík"]), (["India"], ["New Delhi"]), (["Indonesia"], ["Jakarta"]),
    (["Iran"], ["Tehran", "Teheran"]), (["Iraq"], ["Baghdad"]), (["Ireland"], ["Dublin"]), (["Italy"], ["Rome"]),
    (["Jamaica"], ["Kingston"]), (["Japan"], ["Tokyo"]), (["Jordan"], ["Amman"]),
    (["Kazakhstan"], ["Astana", "Nur-Sultan"]), (["Kenya"], ["Nairobi"]), (["Kyrgyzstan"], ["Bishkek"]),
    (["Laos"], ["Vientiane"]), (["Latvia"], ["Riga"]), (["Lebanon"], ["Beirut"]), (["Liberia"], ["Monrovia"]),
    (["Libya"], ["Tripoli"]), (["Lithuania"], ["Vilnius"]), (["Madagascar"], ["Antananarivo"]),
    (["Malawi"], ["Lilongwe"]), (["Malaysia"], ["Kuala Lumpur"]), (["Mali"], ["Bamako"]), (["Malta"], ["Valletta"]),
    (["Mongolia"], ["Ulaanbaatar", "Ulan Bator"]), (["Morocco"], ["Rabat"]), (["Mozambique"], ["Maputo"]),
    (["Nepal"], ["Kathmandu"]), (["the Netherlands", "Netherlands", "Holland"], ["Amsterdam"]),
    (["New Zealand"], ["Wellington"]), (["Nicaragua"], ["Managua"]), (["Nigeria"], ["Abuja"]),
    (["North Korea"], ["Pyongyang"]), (["Norway"], ["Oslo"]), (["Oman"], ["Muscat"]), (["Pakistan"], ["Islamabad"]),
    (["Paraguay"], ["Asunción", "Asuncion"]), (["Peru"], ["Lima"]), (["the Philippines", "Philippines"], ["Manila"]),
    (["Poland"], ["Warsaw"]), (["Portugal"], ["Lisbon"]), (["Qatar"], ["Doha"]), (["Romania"], ["Bucharest"]),
    (["Russia"], ["Moscow"]), (["Rwanda"], ["Kigali"]), (["Saudi Arabia"], ["Riyadh"]), (["Senegal"], ["Dakar"]),
    (["Serbia"], ["Belgrade"]), (["Sierra Leone"], ["Freetown"]), (["Slovakia"], ["Bratislava"]),
    (["Slovenia"], ["Ljubljana"]), (["Somalia"], ["Mogadishu"]), (["South Korea"], ["Seoul"]),
    (["Spain"], ["Madrid"]), (["Sudan"], ["Khartoum"]), (["Sweden"], ["Stockholm"]),
    (["Switzerland"], ["Bern", "Berne"]), (["Syria"], ["Damascus"]), (["Taiwan"], ["Taipei"]),
    (["Tajikistan"], ["Dushanbe"]), (["Thailand"], ["Bangkok"]), (["Turkey"], ["Ankara"]),
    (["Turkmenistan"], ["Ashgabat"]), (["Uganda"], ["Kampala"]), (["Ukraine"], ["Kyiv", "Kiev"]),
    (["the United Kingdom", "United Kingdom", "Britain", "UK"], ["London"]),
    (["the United States", "United States", "USA"], ["Washington"]), (["Uruguay"], ["Montevideo"]),
    (["Uzbekistan"], ["Tashkent"]), (["Venezuela"], ["Caracas"]), (["Vietnam"], ["Hanoi"]),
    (["Yemen"], ["Sanaa", "Sana'a"]), (["Zambia"], ["Lusaka"]), (["Zimbabwe"], ["Harare"]),
    (["Namibia"], ["Windhoek"]), (["Eritrea"], ["Asmara"]), (["Bhutan"], ["Thimphu"]),
    (["Moldova"], ["Chisinau", "Chișinău"]), (["North Macedonia", "Macedonia"], ["Skopje"]),
    (["Montenegro"], ["Podgorica"]), (["Kosovo"], ["Pristina"]), (["Belize"], ["Belmopan"]),
    (["Barbados"], ["Bridgetown"]), (["the Bahamas", "Bahamas"], ["Nassau"]), (["Mauritania"], ["Nouakchott"]),
    (["Togo"], ["Lomé", "Lome"]), (["Lesotho"], ["Maseru"]), (["Gambia"], ["Banjul"]), (["Gabon"], ["Libreville"]),
    (["Liechtenstein"], ["Vaduz"]), (["Samoa"], ["Apia"]), (["Cape Verde"], ["Praia"]),
    (["Haiti"], ["Port-au-Prince"]), (["the Dominican Republic", "Dominican Republic"], ["Santo Domingo"]),
    (["Puerto Rico"], ["San Juan"]), (["Georgia"], ["Tbilisi"]), (["Costa Rica"], ["San José", "San Jose"]),
    (["Israel"], ["Jerusalem"]), (["Tunisia"], ["Tunis"]), (["Algeria"], ["Algiers"]),
    (["Brazil"], ["Brasília", "Brasilia"]),
]

# (country names, cities; the aliases of a city follow its canonical name after "|")
CITIES = [
    (["Germany"], ["Munich|München", "Hamburg", "Frankfurt", "Cologne"]),
    (["France"], ["Marseille|Marseilles", "Lyon|Lyons", "Toulouse", "Bordeaux"]),
    (["Italy"], ["Milan", "Naples", "Turin", "Florence"]),
    (["Spain"], ["Barcelona", "Seville|Sevilla", "Valencia", "Bilbao"]),
    (["Canada"], ["Toronto", "Vancouver", "Montreal|Montréal", "Calgary"]),
    (["Australia"], ["Sydney", "Melbourne", "Brisbane", "Perth"]),
    (["Japan"], ["Osaka", "Kyoto", "Yokohama", "Hiroshima"]),
    (["China"], ["Shanghai", "Guangzhou", "Shenzhen", "Wuhan"]),
    (["India"], ["Mumbai|Bombay", "Bangalore|Bengaluru", "Kolkata|Calcutta", "Chennai|Madras"]),
    (["Brazil"], ["Rio de Janeiro", "São Paulo|Sao Paulo", "Salvador", "Recife"]),
    (["Russia"], ["Novosibirsk", "Yekaterinburg", "Kazan", "Vladivostok"]),
    (["Turkey"], ["Istanbul", "Izmir", "Antalya", "Bursa"]), (["Egypt"], ["Alexandria", "Giza", "Luxor", "Aswan"]),
    (["the Netherlands", "Netherlands", "Holland"], ["Rotterdam", "Utrecht", "Eindhoven"]),
    (["Belgium"], ["Antwerp", "Ghent", "Bruges"]), (["Switzerland"], ["Zurich|Zürich", "Geneva", "Basel"]),
    (["Austria"], ["Salzburg", "Graz", "Innsbruck"]),
    (["Poland"], ["Krakow|Kraków|Cracow", "Gdansk|Gdańsk", "Wroclaw|Wrocław"]),
    (["Sweden"], ["Gothenburg|Göteborg", "Malmö|Malmo", "Uppsala"]),
    (["Norway"], ["Bergen", "Trondheim", "Stavanger"]),
    (["Mexico"], ["Guadalajara", "Monterrey", "Tijuana", "Acapulco"]),
    (["Pakistan"], ["Karachi", "Lahore", "Peshawar"]), (["Nigeria"], ["Lagos", "Kano", "Ibadan"]),
    (["South Africa"], ["Johannesburg", "Durban", "Cape Town"]), (["Israel"], ["Tel Aviv", "Haifa", "Eilat"]),
    (["Vietnam"], ["Ho Chi Minh City|Saigon", "Da Nang", "Hue"]),
    (["Thailand"], ["Chiang Mai", "Phuket", "Pattaya"]), (["Indonesia"], ["Surabaya", "Bandung", "Medan"]),
    (["the Philippines", "Philippines"], ["Cebu", "Davao", "Quezon City"]),
    (["Ireland"], ["Cork", "Galway", "Limerick"]), (["Portugal"], ["Porto|Oporto", "Coimbra", "Braga"]),
    (["Greece"], ["Thessaloniki", "Patras", "Heraklion"]), (["Kenya"], ["Mombasa", "Kisumu", "Nakuru"]),
    (["Morocco"], ["Casablanca", "Marrakech|Marrakesh", "Fez", "Tangier"]),
    (["Iran"], ["Isfahan", "Shiraz", "Tabriz", "Mashhad"]), (["Iraq"], ["Basra", "Mosul", "Erbil"]),
    (["Saudi Arabia"], ["Jeddah", "Mecca", "Medina"]), (["Colombia"], ["Medellín|Medellin", "Cali", "Cartagena"]),
    (["Chile"], ["Valparaíso|Valparaiso", "Concepción|Concepcion"]),
    (["Peru"], ["Cusco|Cuzco", "Arequipa", "Trujillo"]), (["Ukraine"], ["Odessa|Odesa", "Kharkiv|Kharkov", "Lviv"]),
    (["Scotland"], ["Glasgow", "Edinburgh", "Aberdeen"]), (["New Zealand"], ["Auckland", "Christchurch", "Dunedin"]),
    (["South Korea"], ["Busan|Pusan", "Incheon", "Daegu"]),
    (["Argentina"], ["Córdoba|Cordoba", "Rosario", "Mendoza"]), (["Bangladesh"], ["Chittagong", "Khulna", "Sylhet"]),
    (["Ghana"], ["Kumasi", "Tamale"]), (["Denmark"], ["Aarhus", "Odense"]), (["Finland"], ["Tampere", "Turku"]),
    (["Czech Republic", "the Czech Republic", "Czechia"], ["Brno", "Ostrava"]),
    (["Malaysia"], ["Penang", "Johor Bahru", "Malacca"]), (["Taiwan"], ["Kaohsiung", "Taichung"]),
    (["Lebanon"], ["Sidon", "Byblos", "Tyre"]), (["Syria"], ["Aleppo", "Homs", "Latakia"]),
    (["Afghanistan"], ["Kandahar", "Herat", "Mazar-i-Sharif"]), (["Cuba"], ["Varadero", "Camagüey|Camaguey"]),
    (["Wales"], ["Cardiff", "Swansea"]), (["England"], ["Manchester", "Liverpool", "Birmingham", "Leeds"]),
]

WINDOW = 128                                      # an answer is a target up to this many tokens after its subject

# {s} is the subject and {S} the subject with a capital first letter
TEMPLATES = {
    "cap": ["The capital of {s} is", "The capital city of {s} is", "Q: What is the capital of {s}?\nA:",
            "{S} is a country whose capital is", "In {s}, the seat of government is the city of",
            "Capital of {s}:", "The national government of {s} sits in", "Which city is the capital of {s}? It is"],
    "ctry": ["{S} is a city in", "The city of {S} is located in", "Q: In which country is {s}?\nA:",
             "{S} is a major city in the country of", "{S} is one of the largest cities in",
             "Country where {s} is located:", "I flew to {s}, a city in", "Which country is {s} in? It is in"],
}


def units() -> dict:
    """Per relation, one unit per answer: subject names, answer names and the canonical forms used in prompts."""
    cap = [dict(subj=countries, ans=capitals, canon_subj=[countries[0]], canon_ans=capitals[0])
           for countries, capitals in CAPITALS]
    ctry = []
    for countries, cities in CITIES:
        names = [c.split("|") for c in cities]
        ctry.append(dict(subj=[a for n in names for a in n], ans=countries, canon_subj=[n[0] for n in names],
                         canon_ans=countries[0]))
    return {"cap": cap, "ctry": ctry}


def build_prompts(encode) -> dict:
    """{relation: {"groups": answer token groups, "ids": prompt tokens, "g": answer group of each prompt}} for a
    tokenizer given by its text encoder."""
    bare = lambda name: name[4:] if name.startswith("the ") else name  # noqa: E731
    first = lambda name: encode(" " + bare(name))[0]  # noqa: E731
    fill = lambda tpl, s: encode(tpl.format(s=s, S=s[0].upper() + s[1:]))  # noqa: E731
    out = {}
    for rel, us in units().items():
        cand = [dict(answer=u["canon_ans"], group=sorted({first(a) for a in u["ans"]}), subjects=u["canon_subj"])
                for u in us if not (rel == "ctry" and u["canon_ans"].startswith("the "))]
        count = Counter(t for c in cand for t in c["group"])
        for c in cand:                            # a token of two groups stays only where it is the canonical one
            c["group"] = [t for t in c["group"] if t == first(c["answer"]) or count[t] == 1]
        count = Counter(t for c in cand for t in c["group"])
        kept = []
        for c in cand:
            stem = c["answer"].replace("the ", "")[:4].lower()
            subs = [s for s in c["subjects"] if bare(s)[:4].lower() != stem
                    and not any(set(fill(tpl, s)) & set(c["group"]) for tpl in TEMPLATES[rel])]
            if max(count[t] for t in c["group"]) == 1 and len(subs) >= (2 if rel == "ctry" else 1):
                kept.append(dict(c, subjects=subs[:2] if rel == "ctry" else subs))
        prompts = [(fill(tpl, s), g) for tpl in TEMPLATES[rel] for g, c in enumerate(kept) for s in c["subjects"]]
        out[rel] = dict(groups=[c["group"] for c in kept], ids=[p for p, _ in prompts], g=[g for _, g in prompts])
    return out
