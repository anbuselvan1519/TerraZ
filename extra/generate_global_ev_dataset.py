import os
import random
import numpy as np
import pandas as pd

# Set fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

BASE_RDBMS = r"C:\Users\admin\OneDrive\Documents\EV_mineral_exploration\extra\rdbms"
OUTPUT_CSV = r"C:\Users\admin\OneDrive\Documents\EV_mineral_exploration\dataset.csv"

# Target EV Minerals
EV_COMMODITIES = [
    "Lithium", "Cobalt", "Nickel", "Graphite", "Copper",
    "Manganese", "Rare Earth Elements", "Platinum", "Vanadium",
    "Aluminum", "Zinc", "Titanium"
]

COMMOD_MAP = {
    'Lithium': 'Lithium',
    'Cobalt': 'Cobalt',
    'Nickel': 'Nickel',
    'Graphite': 'Graphite',
    'Copper': 'Copper',
    'Manganese': 'Manganese',
    'REE': 'Rare Earth Elements',
    'Rare Earth Elements': 'Rare Earth Elements',
    'Platinum': 'Platinum',
    'PGE': 'Platinum',
    'Vanadium': 'Vanadium',
    'Aluminum': 'Aluminum',
    'Bauxite': 'Aluminum',
    'Zinc': 'Zinc',
    'Titanium': 'Titanium'
}

# 50 Selected Global Jurisdictions (10 Americas, 12 Africa, 12 Asia, 12 Europe, 4 Oceania)
# Each jurisdiction is mapped to region, and authentic metallogenic belts/provinces/districts
COUNTRY_PROFILES = {
    # ---------------- Americas (10) ----------------
    "Chile": {
        "region": "SA",
        "belts": [
            {"state": "Antofagasta", "district": "Atacama Salar Basin", "commod": "Lithium", "lat": -23.50, "lon": -68.30, "radius": 0.5, "mines": ["Salar de Atacama", "Zaldivar", "Negro Francisco"]},
            {"state": "Atacama", "district": "Maricunga Salar Basin", "commod": "Lithium", "lat": -26.90, "lon": -69.25, "radius": 0.4, "mines": ["Maricunga", "Pedernales", "Laguna Verde"]},
            {"state": "Antofagasta", "district": "Central Andes Porphyry Belt", "commod": "Copper", "lat": -22.30, "lon": -68.90, "radius": 0.8, "mines": ["Chuquicamata", "Escondida", "Spence", "El Abra"]},
            {"state": "Tarapaca", "district": "Collahuasi District", "commod": "Copper", "lat": -20.95, "lon": -68.70, "radius": 0.4, "mines": ["Collahuasi", "Quebrada Blanca", "Cerro Colorado"]},
            {"state": "Coquimbo", "district": "Elqui Province", "commod": "Manganese", "lat": -29.90, "lon": -71.20, "radius": 0.6, "mines": ["El Romeral", "Lambert", "Pelicana"]},
            {"state": "Valparaiso", "district": "Putaendo District", "commod": "Cobalt", "lat": -32.55, "lon": -70.70, "radius": 0.5, "mines": ["Carrizal Alto", "Cobaltera", "La Brea"]}
        ]
    },
    "Argentina": {
        "region": "SA",
        "belts": [
            {"state": "Catamarca", "district": "Hombre Muerto Salar", "commod": "Lithium", "lat": -25.45, "lon": -67.05, "radius": 0.4, "mines": ["Fenix", "Sal de Vida", "Hombre Muerto North"]},
            {"state": "Jujuy", "district": "Olaroz-Cauchari Basin", "commod": "Lithium", "lat": -23.65, "lon": -66.75, "radius": 0.4, "mines": ["Olaroz", "Cauchari", "Pozuelos"]},
            {"state": "Salta", "district": "Rincon Salar District", "commod": "Lithium", "lat": -24.05, "lon": -67.35, "radius": 0.4, "mines": ["Salar del Rincon", "Centenario-Ratones", "Pastos Grandes"]},
            {"state": "San Juan", "district": "Frontal Cordillera", "commod": "Copper", "lat": -31.85, "lon": -70.25, "radius": 0.7, "mines": ["Josemaria", "Los Azules", "El Pachon", "Altar"]},
            {"state": "Catamarca", "district": "Farallon Negro District", "commod": "Copper", "lat": -27.30, "lon": -66.60, "radius": 0.5, "mines": ["Bajo de la Alumbrera", "Agua Rica", "Filo del Sol"]},
            {"state": "Jujuy", "district": "Puna Metallogenic Province", "commod": "Zinc", "lat": -22.40, "lon": -65.70, "radius": 0.5, "mines": ["Mina Aguilar", "Pirquitas", "Chorrillos"]}
        ]
    },
    "Bolivia": {
        "region": "SA",
        "belts": [
            {"state": "Potosi", "district": "Uyuni Salar Basin", "commod": "Lithium", "lat": -20.25, "lon": -67.55, "radius": 0.8, "mines": ["Salar de Uyuni", "Rio Grande", "Colcha K", "Coipasa"]},
            {"state": "Potosi", "district": "Southern Bolivian Silver-Zinc Belt", "commod": "Zinc", "lat": -21.15, "lon": -66.30, "radius": 0.7, "mines": ["San Cristobal", "Porco", "Cerro Rico", "Pulacayo"]},
            {"state": "Oruro", "district": "Central Bolivian Tin-Zinc Belt", "commod": "Zinc", "lat": -18.25, "lon": -66.90, "radius": 0.6, "mines": ["Bolivar", "Poopo", "Huanuni"]},
            {"state": "Santa Cruz", "district": "Precambrian Shield", "commod": "Rare Earth Elements", "lat": -16.30, "lon": -61.10, "radius": 0.8, "mines": ["Cerro Manomo", "Rincon del Tigre", "San Ignacio"]},
            {"state": "La Paz", "district": "Cordillera Real", "commod": "Copper", "lat": -16.50, "lon": -68.10, "radius": 0.6, "mines": ["Corocoro", "Chacarilla", "Laurani"]}
        ]
    },
    "Brazil": {
        "region": "SA",
        "belts": [
            {"state": "Minas Gerais", "district": "Aracuai Pegmatite District", "commod": "Lithium", "lat": -16.85, "lon": -42.05, "radius": 0.6, "mines": ["Grota do Cirilo", "Miterra", "Itinga", "CBL"]},
            {"state": "Para", "district": "Carajas Mineral Province", "commod": "Copper", "lat": -6.05, "lon": -50.15, "radius": 0.8, "mines": ["Salobo", "Sossego", "Cristalino", "Alemao"]},
            {"state": "Goias", "district": "Niquelandia Complex", "commod": "Nickel", "lat": -14.45, "lon": -48.45, "radius": 0.5, "mines": ["Niquelandia", "Barro Alto", "Santa Rita"]},
            {"state": "Bahia", "district": "Itabuna-Salvador Belt", "commod": "Graphite", "lat": -14.80, "lon": -39.30, "radius": 0.6, "mines": ["Itapecerica", "Salto da Divisa", "Pedra Preta"]},
            {"state": "Amapa", "district": "Serra do Navio", "commod": "Manganese", "lat": 0.90, "lon": -52.00, "radius": 0.5, "mines": ["Serra do Navio", "Azulao", "Buritirama"]},
            {"state": "Para", "district": "Trombetas Basin", "commod": "Aluminum", "lat": -1.50, "lon": -56.40, "radius": 0.6, "mines": ["Porto Trombetas", "Paragominas", "Juruti"]},
            {"state": "Goias", "district": "Catalao Carbonatite", "commod": "Rare Earth Elements", "lat": -18.15, "lon": -47.80, "radius": 0.4, "mines": ["Catalao I", "Catalao II", "Araxa"]}
        ]
    },
    "Peru": {
        "region": "SA",
        "belts": [
            {"state": "Arequipa", "district": "Southern Peru Porphyry Belt", "commod": "Copper", "lat": -16.50, "lon": -71.50, "radius": 0.7, "mines": ["Cerro Verde", "Cuajone", "Toquepala", "Tia Maria"]},
            {"state": "Ancash", "district": "Cordillera Negra", "commod": "Zinc", "lat": -9.55, "lon": -77.30, "radius": 0.6, "mines": ["Antamina", "Pierina", "Huaron", "Iscaycruz"]},
            {"state": "Apurimac", "district": "Andahuaylas-Yauri Belt", "commod": "Copper", "lat": -14.10, "lon": -72.30, "radius": 0.6, "mines": ["Las Bambas", "Constancia", "Tintaya", "Haquira"]},
            {"state": "Puno", "district": "Macusani Plateau", "commod": "Lithium", "lat": -14.05, "lon": -70.45, "radius": 0.5, "mines": ["Falchani", "Macusani", "Corani"]},
            {"state": "Pasco", "district": "Cerro de Pasco Basin", "commod": "Zinc", "lat": -10.65, "lon": -76.25, "radius": 0.5, "mines": ["Cerro de Pasco", "Atacocha", "Milpo", "El Brocal"]}
        ]
    },
    "Canada": {
        "region": "NA",
        "belts": [
            {"state": "Ontario", "district": "Sudbury Basin", "commod": "Nickel", "lat": 46.50, "lon": -81.00, "radius": 0.6, "mines": ["Creighton", "Totten", "Coleman", "Copper Cliff"]},
            {"state": "Quebec", "district": "James Bay Pegmatite Belt", "commod": "Lithium", "lat": 51.80, "lon": -76.50, "radius": 1.2, "mines": ["Whabouchi", "James Bay", "Rose", "Moblan"]},
            {"state": "Manitoba", "district": "Thompson Nickel Belt", "commod": "Cobalt", "lat": 55.75, "lon": -97.85, "radius": 0.8, "mines": ["Thompson", "Birchtree", "Pipe", "Bucko Lake"]},
            {"state": "Ontario", "district": "Lac des Iles Complex", "commod": "Platinum", "lat": 49.15, "lon": -89.60, "radius": 0.6, "mines": ["Lac des Iles", "Marathon", "River Valley"]},
            {"state": "Quebec", "district": "Grenville Province", "commod": "Graphite", "lat": 46.20, "lon": -75.10, "radius": 0.8, "mines": ["Lac des Iles Graphite", "Matawinie", "Okanagan"]},
            {"state": "British Columbia", "district": "Highland Valley", "commod": "Copper", "lat": 50.45, "lon": -121.00, "radius": 0.8, "mines": ["Highland Valley", "Gibraltar", "Mt Milligan", "Red Chris"]},
            {"state": "Northwest Territories", "district": "Nechalacho Alkaline Complex", "commod": "Rare Earth Elements", "lat": 62.10, "lon": -112.60, "radius": 0.5, "mines": ["Nechalacho", "Thor Lake", "Hoidas Lake"]}
        ]
    },
    "United States": {
        "region": "NA",
        "belts": [
            {"state": "Nevada", "district": "McDermitt Caldera", "commod": "Lithium", "lat": 41.70, "lon": -117.90, "radius": 0.5, "mines": ["Thacker Pass", "Montana Mountains", "Opalite"]},
            {"state": "Nevada", "district": "Clayton Valley", "commod": "Lithium", "lat": 37.75, "lon": -117.60, "radius": 0.4, "mines": ["Silver Peak", "Clayton Valley Deep", "Albemarle Brine"]},
            {"state": "Arizona", "district": "Globe-Miami District", "commod": "Copper", "lat": 33.40, "lon": -110.85, "radius": 0.7, "mines": ["Morenci", "Ray", "Bagdad", "Pinto Valley"]},
            {"state": "California", "district": "Mountain Pass Complex", "commod": "Rare Earth Elements", "lat": 35.48, "lon": -115.53, "radius": 0.4, "mines": ["Mountain Pass", "Sulfur Springs", "Mescal"]},
            {"state": "Montana", "district": "Stillwater Complex", "commod": "Platinum", "lat": 45.38, "lon": -109.88, "radius": 0.5, "mines": ["Stillwater", "East Boulder", "Blitz"]},
            {"state": "Minnesota", "district": "Duluth Complex", "commod": "Nickel", "lat": 47.75, "lon": -91.80, "radius": 0.6, "mines": ["NorthMet", "Twin Metals", "Maturin", "Tamarack"]},
            {"state": "Idaho", "district": "Idaho Cobalt Belt", "commod": "Cobalt", "lat": 45.10, "lon": -114.30, "radius": 0.5, "mines": ["Blackbird", "Ram", "Iron Creek"]}
        ]
    },
    "Mexico": {
        "region": "NA",
        "belts": [
            {"state": "Sonora", "district": "Bacadéhuachi Basin", "commod": "Lithium", "lat": 29.80, "lon": -109.15, "radius": 0.5, "mines": ["Sonora Lithium", "La Ventana", "El Sauz"]},
            {"state": "Sonora", "district": "Cananea District", "commod": "Copper", "lat": 30.98, "lon": -110.30, "radius": 0.6, "mines": ["Buenavista del Cobre", "La Caridad", "Pilares"]},
            {"state": "Zacatecas", "district": "Fresnillo District", "commod": "Zinc", "lat": 23.18, "lon": -102.87, "radius": 0.6, "mines": ["Penasquito", "Fresnillo", "Saucito", "Tayahua"]},
            {"state": "Baja California Sur", "district": "Santa Rosalia", "commod": "Cobalt", "lat": 27.35, "lon": -112.28, "radius": 0.5, "mines": ["El Boleo", "San Luciano", "Purgatorio"]},
            {"state": "Hidalgo", "district": "Molango District", "commod": "Manganese", "lat": 20.80, "lon": -98.75, "radius": 0.5, "mines": ["Molango", "Tetzintla", "Naopa"]}
        ]
    },
    "Colombia": {
        "region": "SA",
        "belts": [
            {"state": "Cordoba", "district": "Cerromatoso Laterite Belt", "commod": "Nickel", "lat": 7.95, "lon": -75.55, "radius": 0.5, "mines": ["Cerromatoso", "Montelibano", "Ure"]},
            {"state": "Antioquia", "district": "Middle Cauca Belt", "commod": "Copper", "lat": 6.20, "lon": -75.70, "radius": 0.6, "mines": ["Quebradona", "Buritica", "Titiribi"]},
            {"state": "Putumayo", "district": "Eastern Cordillera", "commod": "Copper", "lat": 1.15, "lon": -76.65, "radius": 0.5, "mines": ["Mocoa", "San Francisco", "Villagarzon"]},
            {"state": "Santander", "district": "California-Vetas District", "commod": "Zinc", "lat": 7.35, "lon": -73.00, "radius": 0.5, "mines": ["La Bodega", "Angostura", "California"]},
            {"state": "Vichada", "district": "Guiana Shield", "commod": "Titanium", "lat": 4.40, "lon": -69.80, "radius": 0.8, "mines": ["Cumaribo", "Santa Rita", "Mataven"]}
        ]
    },
    "Ecuador": {
        "region": "SA",
        "belts": [
            {"state": "Zamora Chinchipe", "district": "Cordillera del Condor", "commod": "Copper", "lat": -3.55, "lon": -78.50, "radius": 0.5, "mines": ["Mirador", "Fruta del Norte", "San Carlos-Panantza"]},
            {"state": "Imbabura", "district": "Northern Andes Porphyry Belt", "commod": "Copper", "lat": 0.60, "lon": -78.20, "radius": 0.5, "mines": ["Cascabel", "Llurimagua", "Alpala"]},
            {"state": "Morona Santiago", "district": "Warintza District", "commod": "Cobalt", "lat": -3.15, "lon": -78.25, "radius": 0.4, "mines": ["Warintza Central", "Warintza West", "Yawi"]},
            {"state": "Azuay", "district": "Southern Sierra", "commod": "Zinc", "lat": -3.00, "lon": -79.20, "radius": 0.5, "mines": ["Loma Larga", "Rio Blanco", "San Gerardo"]}
        ]
    },

    # ---------------- Africa (12) ----------------
    "DR Congo": {
        "region": "AF",
        "belts": [
            {"state": "Lualaba", "district": "Kolwezi Nappe District", "commod": "Cobalt", "lat": -10.72, "lon": 25.47, "radius": 0.5, "mines": ["Kamoto", "Mutanda", "KOV", "Mutoshi", "Mashamba"]},
            {"state": "Haut-Katanga", "district": "Tenke Fungurume Belt", "commod": "Cobalt", "lat": -10.58, "lon": 26.18, "radius": 0.6, "mines": ["Tenke", "Fungurume", "Kisanfu", "Kinsenda"]},
            {"state": "Lualaba", "district": "Kamoa-Kakula Corridor", "commod": "Copper", "lat": -10.85, "lon": 25.20, "radius": 0.5, "mines": ["Kamoa", "Kakula", "Makalu", "Kansoko"]},
            {"state": "Tanganyika", "district": "Manono Pegmatite Field", "commod": "Lithium", "lat": -7.30, "lon": 27.42, "radius": 0.5, "mines": ["Roche Dure", "Manono Central", "Mpete", "Kahungwe"]},
            {"state": "Haut-Katanga", "district": "Kipushi Fault Corridor", "commod": "Zinc", "lat": -11.75, "lon": 27.25, "radius": 0.4, "mines": ["Kipushi Big Zinc", "Kafubu", "Ruashi"]},
            {"state": "North Kivu", "district": "Walikale Province", "commod": "Rare Earth Elements", "lat": -1.40, "lon": 28.05, "radius": 0.6, "mines": ["Bisie", "Mpama North", "Lueshe"]}
        ]
    },
    "South Africa": {
        "region": "AF",
        "belts": [
            {"state": "Limpopo", "district": "Bushveld Western Limb", "commod": "Platinum", "lat": -24.20, "lon": 29.30, "radius": 0.8, "mines": ["Mogalakwena", "Amandelbult", "Bafokeng Rasimone", "Marikana"]},
            {"state": "Mpumalanga", "district": "Bushveld Eastern Limb", "commod": "Platinum", "lat": -24.80, "lon": 30.15, "radius": 0.8, "mines": ["Modikwa", "Two Rivers", "Mototolo", "Marula"]},
            {"state": "Northern Cape", "district": "Kalahari Manganese Field", "commod": "Manganese", "lat": -27.20, "lon": 22.95, "radius": 0.6, "mines": ["Mamatwan", "Wessels", "Nchwaning", "Gloria", "Tshipi Borwa"]},
            {"state": "Mpumalanga", "district": "Bushveld Vanadiferous Titanomagnetite", "commod": "Vanadium", "lat": -25.40, "lon": 29.90, "radius": 0.6, "mines": ["Rhovan", "Vametco", "Steelpoortdrif", "Brits"]},
            {"state": "Limpopo", "district": "Palabora Igneous Complex", "commod": "Copper", "lat": -23.95, "lon": 31.15, "radius": 0.4, "mines": ["Palabora", "Foskor", "Phalaborwa North"]},
            {"state": "Northern Cape", "district": "Aggeneys-Gamsberg Belt", "commod": "Zinc", "lat": -29.25, "lon": 18.95, "radius": 0.5, "mines": ["Gamsberg", "Black Mountain", "Broken Hill"]}
        ]
    },
    "Zimbabwe": {
        "region": "AF",
        "belts": [
            {"state": "Masvingo", "district": "Bikita Pegmatite Belt", "commod": "Lithium", "lat": -19.95, "lon": 31.43, "radius": 0.4, "mines": ["Bikita Pegmatite", "Al Hayat", "Shaft 1"]},
            {"state": "Mashonaland East", "district": "Arcadia Lithium Corridor", "commod": "Lithium", "lat": -17.75, "lon": 31.40, "radius": 0.4, "mines": ["Arcadia", "Goromonzi", "Step Aside"]},
            {"state": "Matabeleland South", "district": "Fort Rixon Pegmatite Field", "commod": "Lithium", "lat": -20.05, "lon": 29.35, "radius": 0.4, "mines": ["Zulu Lithium", "Kamativi", "Sandawana"]},
            {"state": "Mashonaland West", "district": "Great Dyke Intrusion", "commod": "Platinum", "lat": -18.25, "lon": 30.25, "radius": 0.9, "mines": ["Mimosa", "Zimplats Ngezi", "Unki", "Darwendale"]},
            {"state": "Midlands", "district": "Great Dyke Ultramafics", "commod": "Nickel", "lat": -19.10, "lon": 29.90, "radius": 0.6, "mines": ["Trojan", "Hunters Road", "Ephraim"]}
        ]
    },
    "Zambia": {
        "region": "AF",
        "belts": [
            {"state": "Copperbelt", "district": "Central Zambian Copperbelt", "commod": "Copper", "lat": -12.80, "lon": 28.20, "radius": 0.7, "mines": ["Nkana", "Mufulira", "Konkola", "Chibuluma", "Chambishi"]},
            {"state": "North-Western", "district": "Solwezi Domes Region", "commod": "Copper", "lat": -12.18, "lon": 26.40, "radius": 0.6, "mines": ["Kansanshi", "Lumwana", "Sentinel", "Enterprise"]},
            {"state": "Copperbelt", "district": "Chambishi Basin", "commod": "Cobalt", "lat": -12.65, "lon": 28.05, "radius": 0.5, "mines": ["Chambishi Cobalt", "Baluba", "Luanshya"]},
            {"state": "Central", "district": "Kabwe Pb-Zn Belt", "commod": "Zinc", "lat": -14.45, "lon": 28.45, "radius": 0.4, "mines": ["Kabwe", "Broken Hill Zambia", "Luiri Hill"]},
            {"state": "Southern", "district": "Kariba Pegmatite Belt", "commod": "Lithium", "lat": -16.80, "lon": 27.10, "radius": 0.5, "mines": ["Choma Pegmatite", "Sinazongwe", "Batoka"]}
        ]
    },
    "Guinea": {
        "region": "AF",
        "belts": [
            {"state": "Boke", "district": "Boke Bauxite Plateau", "commod": "Aluminum", "lat": 10.95, "lon": -14.30, "radius": 0.8, "mines": ["Sangaredi", "Dian-Dian", "Kamsar", "SMB-Winning"]},
            {"state": "Kindia", "district": "Kindia Plateau", "commod": "Aluminum", "lat": 10.05, "lon": -12.85, "radius": 0.6, "mines": ["Friguia", "Debele", "Kindia North"]},
            {"state": "Faranah", "district": "Dabola Bauxite Belt", "commod": "Aluminum", "lat": 10.75, "lon": -11.10, "radius": 0.6, "mines": ["Dabola", "Tougue", "Bantignel"]},
            {"state": "Nzerekore", "district": "Simandou Range", "commod": "Cobalt", "lat": 8.55, "lon": -8.90, "radius": 0.5, "mines": ["Simandou Blocks 1-2", "Nimba", "Zogota"]},
            {"state": "Kankan", "district": "Siguiri Basin", "commod": "Graphite", "lat": 11.40, "lon": -9.20, "radius": 0.6, "mines": ["Siguiri Graphite", "Kouroussa", "Mandiana"]}
        ]
    },
    "Mozambique": {
        "region": "AF",
        "belts": [
            {"state": "Cabo Delgado", "district": "Balama Flake Graphite Belt", "commod": "Graphite", "lat": -13.35, "lon": 38.55, "radius": 0.6, "mines": ["Balama Syrah", "Ancuabe", "Nipurn", "Nicanda Hill"]},
            {"state": "Nampula", "district": "Heavy Mineral Coast", "commod": "Titanium", "lat": -16.20, "lon": 39.80, "radius": 0.8, "mines": ["Moma Kenmare", "Corridor Sands", "Pebane", "Angoche"]},
            {"state": "Zambezia", "district": "Alto Ligonha Pegmatite Province", "commod": "Rare Earth Elements", "lat": -15.80, "lon": 38.20, "radius": 0.7, "mines": ["Alto Ligonha", "Muiane", "Marropino"]},
            {"state": "Tete", "district": "Tete Igneous Complex", "commod": "Vanadium", "lat": -15.70, "lon": 33.60, "radius": 0.6, "mines": ["Tete Titanomagnetite", "Moatize South", "Chiuta"]},
            {"state": "Manica", "district": "Manica Greenstone Belt", "commod": "Copper", "lat": -18.95, "lon": 32.85, "radius": 0.5, "mines": ["Mundonguara", "Chimoio", "Revue"]}
        ]
    },
    "Madagascar": {
        "region": "AF",
        "belts": [
            {"state": "Alaotra-Mangoro", "district": "Ambatovy Laterite Belt", "commod": "Nickel", "lat": -18.85, "lon": 48.30, "radius": 0.5, "mines": ["Ambatovy", "Analamay", "Moramanga"]},
            {"state": "Atsinanana", "district": "Eastern Madagascar Graphite Belt", "commod": "Graphite", "lat": -18.15, "lon": 48.70, "radius": 0.7, "mines": ["Molo", "Toamasina", "Sahamamy", "Antsirakambo"]},
            {"state": "Atsimo-Andrefana", "district": "Ampanihy Metamorphic Belt", "commod": "Graphite", "lat": -24.70, "lon": 44.75, "radius": 0.6, "mines": ["Ampanihy", "Fotadrevo", "Greenwing"]},
            {"state": "Diana", "district": "Ambohimirahavavy Alkaline Complex", "commod": "Rare Earth Elements", "lat": -13.50, "lon": 48.40, "radius": 0.5, "mines": ["Tantalus REE", "Ampasindava", "Antsiranana"]},
            {"state": "Anosy", "district": "Tolagnaro Coastal Sands", "commod": "Titanium", "lat": -25.00, "lon": 46.95, "radius": 0.5, "mines": ["QMM Mandena", "Sainte Luce", "Petriky"]}
        ]
    },
    "Morocco": {
        "region": "AF",
        "belts": [
            {"state": "Draa-Tafilalet", "district": "Bou Azzer Inlier", "commod": "Cobalt", "lat": 30.52, "lon": -6.90, "radius": 0.4, "mines": ["Bou Azzer", "Aghbar", "Filon 7", "Tamdrost"]},
            {"state": "Souss-Massa", "district": "Anti-Atlas Copper Belt", "commod": "Copper", "lat": 29.80, "lon": -8.80, "radius": 0.6, "mines": ["Bleida", "Tiouit", "Akka", "Tizert"]},
            {"state": "Oriental", "district": "Touissit-Bou Beker District", "commod": "Zinc", "lat": 34.45, "lon": -1.75, "radius": 0.5, "mines": ["Touissit", "Bou Beker", "Sidi Lahcen"]},
            {"state": "Marrakech-Safi", "district": "Guemassa Basin", "commod": "Manganese", "lat": 31.35, "lon": -8.20, "radius": 0.5, "mines": ["Imini", "Hajjar", "Draa Sfar"]},
            {"state": "Draa-Tafilalet", "district": "Imiter District", "commod": "Cobalt", "lat": 31.35, "lon": -5.85, "radius": 0.5, "mines": ["Imiter", "Bou Madine", "Tichka"]}
        ]
    },
    "Ghana": {
        "region": "AF",
        "belts": [
            {"state": "Central", "district": "Mfantseman Pegmatite Belt", "commod": "Lithium", "lat": 5.30, "lon": -1.05, "radius": 0.4, "mines": ["Ewoyaa", "Aboadze", "Saltpond"]},
            {"state": "Western", "district": "Sefwi Bauxite Belt", "commod": "Aluminum", "lat": 6.25, "lon": -2.35, "radius": 0.6, "mines": ["Awaso", "Nyinahin", "Kyebi"]},
            {"state": "Western", "district": "Nsuta Basin", "commod": "Manganese", "lat": 5.28, "lon": -1.98, "radius": 0.4, "mines": ["Nsuta Manganese", "Tarkwa Manganese", "Prestea"]},
            {"state": "Ashanti", "district": "Ashanti Metamorphic Belt", "commod": "Graphite", "lat": 6.45, "lon": -1.50, "radius": 0.6, "mines": ["Kumasi Graphite", "Obuasi North", "Konongo"]},
            {"state": "Eastern", "district": "Kibi Bauxite Belt", "commod": "Aluminum", "lat": 6.15, "lon": -0.55, "radius": 0.5, "mines": ["Atewa Range", "Kibi Plateau", "Asiakwa"]}
        ]
    },
    "Namibia": {
        "region": "AF",
        "belts": [
            {"state": "Erongo", "district": "Damara Orogen Pegmatites", "commod": "Lithium", "lat": -21.40, "lon": 14.80, "radius": 0.6, "mines": ["Uis", "Karibib", "Helikon", "Rubicon"]},
            {"state": "Oshikoto", "district": "Otavi Mountainland", "commod": "Zinc", "lat": -19.25, "lon": 17.70, "radius": 0.6, "mines": ["Tsumeb", "Kombat", "Berg Aukas", "Abenab"]},
            {"state": "Karas", "district": "Rosh Pinah Belt", "commod": "Zinc", "lat": -27.95, "lon": 16.75, "radius": 0.5, "mines": ["Rosh Pinah", "Skorpion Zinc", "Gergarub"]},
            {"state": "Kunene", "district": "Opuwo Cobalt-Copper Belt", "commod": "Cobalt", "lat": -18.05, "lon": 13.85, "radius": 0.6, "mines": ["Opuwo", "Kunene Project", "Kaoko"]},
            {"state": "Otjozondjupa", "district": "Otjiwarongo Carbonatite", "commod": "Rare Earth Elements", "lat": -20.50, "lon": 16.80, "radius": 0.5, "mines": ["Lofdal", "Okorusu", "Eureka"]}
        ]
    },
    "Tanzania": {
        "region": "AF",
        "belts": [
            {"state": "Kagera", "district": "Kabanga Mafic-Ultramafic Belt", "commod": "Nickel", "lat": -2.90, "lon": 30.50, "radius": 0.4, "mines": ["Kabanga Nickel", "Kapalagulu", "Ngara"]},
            {"state": "Lindi", "district": "Ruangwa Metamorphic Belt", "commod": "Graphite", "lat": -10.05, "lon": 38.95, "radius": 0.6, "mines": ["Chilalo", "Nachu", "Lindi Jumbo"]},
            {"state": "Morogoro", "district": "Mahenge Graphite Field", "commod": "Graphite", "lat": -8.70, "lon": 36.70, "radius": 0.5, "mines": ["Epanko", "Mahenge Black Rock", "Ulanga"]},
            {"state": "Songwe", "district": "Panda Hill Carbonatite", "commod": "Rare Earth Elements", "lat": -8.98, "lon": 33.23, "radius": 0.4, "mines": ["Panda Hill", "Ngualla", "Songwe Hill"]},
            {"state": "Mara", "district": "Lake Victoria Greenstone", "commod": "Copper", "lat": -1.50, "lon": 34.00, "radius": 0.6, "mines": ["North Mara", "Bulyanhulu Copper", "Geita Cu"]}
        ]
    },
    "Mali": {
        "region": "AF",
        "belts": [
            {"state": "Sikasso", "district": "Bougouni Pegmatite Belt", "commod": "Lithium", "lat": 11.40, "lon": -7.50, "radius": 0.5, "mines": ["Goulamina", "Bougouni Leo", "Kodal", "Torakoro"]},
            {"state": "Kayes", "district": "Kenieba Inlier", "commod": "Cobalt", "lat": 12.85, "lon": -11.20, "radius": 0.6, "mines": ["Falea", "Kenieba", "Loulo Cobalt"]},
            {"state": "Sikasso", "district": "Yanfolila Basin", "commod": "Aluminum", "lat": 11.15, "lon": -8.30, "radius": 0.5, "mines": ["Bale Bauxite", "Kita Plateau", "Bafing"]},
            {"state": "Koulikoro", "district": "Kangaba Manganese Field", "commod": "Manganese", "lat": 11.95, "lon": -8.40, "radius": 0.5, "mines": ["Ansongo", "Tinkisso", "Kangaba"]}
        ]
    },

    # ---------------- Asia (12) ----------------
    "China": {
        "region": "AS",
        "belts": [
            {"state": "Inner Mongolia", "district": "Bayan Obo Complex", "commod": "Rare Earth Elements", "lat": 41.78, "lon": 109.97, "radius": 0.5, "mines": ["Bayan Obo Main", "Bayan Obo East", "Damao"]},
            {"state": "Sichuan", "district": "Songpan-Ganze Pegmatite Belt", "commod": "Lithium", "lat": 30.95, "lon": 101.95, "radius": 0.7, "mines": ["Jiajika", "Maerkang", "Lijiagou", "Yajiang"]},
            {"state": "Qinghai", "district": "Qaidam Salar Basin", "commod": "Lithium", "lat": 37.05, "lon": 95.30, "radius": 0.9, "mines": ["Qarhan Salar", "Dongtai Jinai'er", "Xitai Jinai'er", "Yiliping"]},
            {"state": "Jiangxi", "district": "Yichun Pegmatite Field", "commod": "Lithium", "lat": 27.80, "lon": 114.40, "radius": 0.5, "mines": ["Yichun 414", "Tantalum-Lithium Mine", "Xinfeng"]},
            {"state": "Gansu", "district": "Jinchuan Ultramafic Intrusion", "commod": "Nickel", "lat": 38.50, "lon": 102.18, "radius": 0.4, "mines": ["Jinchuan No. 1", "Jinchuan No. 2", "Jinchuan No. 3"]},
            {"state": "Heilongjiang", "district": "Luobei Graphite Belt", "commod": "Graphite", "lat": 47.58, "lon": 130.82, "radius": 0.5, "mines": ["Luobei Yunshan", "Jixi", "Liumao"]},
            {"state": "Jiangxi", "district": "Dexing Porphyry Cluster", "commod": "Copper", "lat": 28.95, "lon": 117.72, "radius": 0.5, "mines": ["Dexing", "Tongkuangyu", "Yulong", "Zijinshan"]},
            {"state": "Guizhou", "district": "Songtao Manganese Basin", "commod": "Manganese", "lat": 28.15, "lon": 109.20, "radius": 0.6, "mines": ["Songtao", "Datangpo", "Xiushan", "Taiping"]}
        ]
    },
    "Indonesia": {
        "region": "AS",
        "belts": [
            {"state": "Central Sulawesi", "district": "Morowali Ultramafic Complex", "commod": "Nickel", "lat": -2.85, "lon": 122.15, "radius": 0.6, "mines": ["Morowali IMIP", "Bahodopi", "Kolonodale", "Bungku"]},
            {"state": "South Sulawesi", "district": "Sorowako Laterite Plateau", "commod": "Nickel", "lat": -2.55, "lon": 121.35, "radius": 0.5, "mines": ["Sorowako Vale", "Petea", "Matano"]},
            {"state": "North Maluku", "district": "Weda Bay Halmahera", "commod": "Nickel", "lat": 0.45, "lon": 127.95, "radius": 0.6, "mines": ["Weda Bay IWIP", "Gebe Island", "Pakal Island", "Maba"]},
            {"state": "Southeast Sulawesi", "district": "Pomalaa Laterite District", "commod": "Cobalt", "lat": -4.18, "lon": 121.60, "radius": 0.5, "mines": ["Pomalaa Antam", "Kabaena", "Konawe"]},
            {"state": "West Papua", "district": "Gag & Waigeo Islands", "commod": "Cobalt", "lat": -0.45, "lon": 129.85, "radius": 0.5, "mines": ["Gag Island", "Waigeo", "Manuran"]},
            {"state": "Papua", "district": "Grasberg Igneous Complex", "commod": "Copper", "lat": -4.05, "lon": 137.11, "radius": 0.4, "mines": ["Grasberg Open Pit", "Deep Mill Level Zone", "Big Gossan"]},
            {"state": "West Kalimantan", "district": "Ketapang Bauxite Belt", "commod": "Aluminum", "lat": -1.85, "lon": 110.00, "radius": 0.7, "mines": ["Tayan", "Kendawangan", "Memapawah"]}
        ]
    },
    "India": {
        "region": "AS",
        "belts": [
            {"state": "Odisha", "district": "Sukinda Ultramafic Valley", "commod": "Nickel", "lat": 20.95, "lon": 85.80, "radius": 0.5, "mines": ["Sukinda Chrome-Nickel", "Kalyanpur", "Kansa"]},
            {"state": "Odisha", "district": "East Coast Bauxite Belt", "commod": "Aluminum", "lat": 18.85, "lon": 83.00, "radius": 0.7, "mines": ["Panchpatmali", "Baphlimali", "Kodingamali", "Gandhamardan"]},
            {"state": "Madhya Pradesh", "district": "Balaghat Manganese Belt", "commod": "Manganese", "lat": 21.80, "lon": 80.20, "radius": 0.6, "mines": ["Bharweli", "Ukwa", "Dongri Buzurg", "Mansar"]},
            {"state": "Rajasthan", "district": "Khetri Copper Belt", "commod": "Copper", "lat": 28.00, "lon": 75.80, "radius": 0.5, "mines": ["Madhan-Kudhan", "Kolihan", "Chandmari"]},
            {"state": "Rajasthan", "district": "Zawar-Rajpura Dariba", "commod": "Zinc", "lat": 24.35, "lon": 73.70, "radius": 0.5, "mines": ["Rampura Agucha", "Sindesar Khurd", "Zawar", "Rajpura Dariba"]},
            {"state": "Jammu and Kashmir", "district": "Reasi Salal-Haimana", "commod": "Lithium", "lat": 33.10, "lon": 74.85, "radius": 0.4, "mines": ["Salal Lithium", "Haimana", "Katra"]},
            {"state": "Kerala", "district": "Chavara Coastal Sands", "commod": "Titanium", "lat": 9.00, "lon": 76.55, "radius": 0.5, "mines": ["Chavara KMML", "Manavalakurichi", "Kollam"]}
        ]
    },
    "Philippines": {
        "region": "AS",
        "belts": [
            {"state": "Surigao del Norte", "district": "Surigao Ophiolite Belt", "commod": "Nickel", "lat": 9.55, "lon": 125.60, "radius": 0.5, "mines": ["Taganito HPAL", "Hinatuan", "Cagdianao", "Adlay"]},
            {"state": "Palawan", "district": "Palawan Ophiolite Belt", "commod": "Nickel", "lat": 9.25, "lon": 118.10, "radius": 0.6, "mines": ["Rio Tuba HPAL", "Berong", "Bataraza"]},
            {"state": "Zambales", "district": "Zambales Ophiolite Complex", "commod": "Cobalt", "lat": 15.35, "lon": 119.95, "radius": 0.5, "mines": ["Acoje", "Coto", "Sta Cruz"]},
            {"state": "South Cotabato", "district": "Tampakan Porphyry Cluster", "commod": "Copper", "lat": 6.45, "lon": 125.05, "radius": 0.4, "mines": ["Tampakan", "King-king", "Boyongan"]},
            {"state": "Benguet", "district": "Mankayan Copper District", "commod": "Copper", "lat": 16.85, "lon": 120.80, "radius": 0.4, "mines": ["Padcal Philex", "Lepanto", "Santo Tomas"]}
        ]
    },
    "Vietnam": {
        "region": "AS",
        "belts": [
            {"state": "Lao Cai", "district": "Red River Metallogenic Belt", "commod": "Copper", "lat": 22.45, "lon": 103.95, "radius": 0.5, "mines": ["Sin Quyen", "Vi Kem", "Bat Xat"]},
            {"state": "Lai Chau", "district": "Dong Pao Alkaline Complex", "commod": "Rare Earth Elements", "lat": 22.25, "lon": 103.55, "radius": 0.4, "mines": ["Dong Pao", "Nam Xe", "Bac Nam Xe"]},
            {"state": "Dak Nong", "district": "Central Highlands Bauxite Plateau", "commod": "Aluminum", "lat": 12.00, "lon": 107.70, "radius": 0.7, "mines": ["Nhan Co", "Tan Rai", "Dak Nong Central"]},
            {"state": "Thai Nguyen", "district": "Nui Phao Skarn Complex", "commod": "Vanadium", "lat": 21.60, "lon": 105.75, "radius": 0.4, "mines": ["Nui Phao", "Dai Tu", "Phan Me"]},
            {"state": "Yen Bai", "district": "Yen Bai Graphite Belt", "commod": "Graphite", "lat": 21.75, "lon": 104.85, "radius": 0.5, "mines": ["Bao Ha", "Tran Yen", "Luc Yen"]}
        ]
    },
    "Kazakhstan": {
        "region": "AS",
        "belts": [
            {"state": "Karaganda", "district": "Zhezkazgan Basin", "commod": "Copper", "lat": 47.80, "lon": 67.70, "radius": 0.7, "mines": ["Zhezkazgan Central", "Annensky", "Stepnoy", "Satpayev"]},
            {"state": "Pavlodar", "district": "Bozshakol Porphyry Cluster", "commod": "Copper", "lat": 52.00, "lon": 74.85, "radius": 0.6, "mines": ["Bozshakol", "Nurkazgan", "Aktogay", "Kounrad"]},
            {"state": "East Kazakhstan", "district": "Rudny Altai VMS Belt", "commod": "Zinc", "lat": 50.35, "lon": 82.60, "radius": 0.6, "mines": ["Maleevsky", "Tishinsky", "Ridder-Sokolny", "Zyryanovsk"]},
            {"state": "Kyzylorda", "district": "Karatau Vanadium Shales", "commod": "Vanadium", "lat": 43.60, "lon": 67.50, "radius": 0.6, "mines": ["Balasausqandiq", "Kurumsak", "Zhabagly"]},
            {"state": "Kostanay", "district": "Lisakovsk Bauxite-Iron", "commod": "Aluminum", "lat": 52.50, "lon": 62.50, "radius": 0.6, "mines": ["Krasnooktyabrsk", "Ayat", "Arkalyk"]}
        ]
    },
    "Uzbekistan": {
        "region": "AS",
        "belts": [
            {"state": "Tashkent", "district": "Almalyk Porphyry District", "commod": "Copper", "lat": 40.85, "lon": 69.60, "radius": 0.5, "mines": ["Kalmakyr", "Yoshlik I", "Sary-Cheku", "Dalnee"]},
            {"state": "Navoiy", "district": "Kyzylkum Metallogenic Belt", "commod": "Vanadium", "lat": 41.50, "lon": 64.20, "radius": 0.8, "mines": ["Muruntau Vanadate", "Besopan", "Dzhetym"]},
            {"state": "Jizzakh", "district": "Nuratau-Turkestan Belt", "commod": "Lithium", "lat": 40.20, "lon": 67.50, "radius": 0.6, "mines": ["Shavazsay", "Tubapet", "Uchkulach Zinc"]},
            {"state": "Kashkadarya", "district": "Chakyl-Kalyan Belt", "commod": "Zinc", "lat": 39.00, "lon": 66.50, "radius": 0.6, "mines": ["Khandiza", "Aktepe", "Kugitang"]}
        ]
    },
    "Turkey": {
        "region": "AS",
        "belts": [
            {"state": "Artvin", "district": "Pontide Metallogenic Belt", "commod": "Copper", "lat": 41.20, "lon": 41.80, "radius": 0.6, "mines": ["Cayeli", "Murgul", "Cerattepe"]},
            {"state": "Erzincan", "district": "Tethyan Porphyry Arc", "commod": "Copper", "lat": 39.45, "lon": 38.50, "radius": 0.6, "mines": ["Copler", "Halilaga", "Kisladag"]},
            {"state": "Manisa", "district": "Gordes Laterite Complex", "commod": "Nickel", "lat": 38.90, "lon": 28.30, "radius": 0.5, "mines": ["Gordes", "Caldag", "Muratdere"]},
            {"state": "Eskisehir", "district": "Beylikova Carbonatite", "commod": "Rare Earth Elements", "lat": 39.70, "lon": 31.20, "radius": 0.4, "mines": ["Kizilcaoren", "Beylikova", "Sivrihisar"]},
            {"state": "Konya", "district": "Seydisehir Bauxite Belt", "commod": "Aluminum", "lat": 37.40, "lon": 31.85, "radius": 0.5, "mines": ["Seydisehir", "Mortas", "Dogankuzu"]}
        ]
    },
    "Mongolia": {
        "region": "AS",
        "belts": [
            {"state": "Omnogovi", "district": "South Gobi Copper Belt", "commod": "Copper", "lat": 43.00, "lon": 106.90, "radius": 0.8, "mines": ["Oyu Tolgoi", "Hugo North", "Heruga", "Oyut"]},
            {"state": "Dornogovi", "district": "Tsagaan Suvarga Corridor", "commod": "Copper", "lat": 43.85, "lon": 108.35, "radius": 0.6, "mines": ["Tsagaan Suvarga", "Shuut", "Mandakh"]},
            {"state": "Orkhon", "district": "Erdenet Metallogenic Province", "commod": "Copper", "lat": 49.05, "lon": 104.10, "radius": 0.5, "mines": ["Erdenetiin Ovoo", "Central Pit", "Oyut Tolgoi Erdenet"]},
            {"state": "Uvs", "district": "Khovd Carbonatite Arc", "commod": "Rare Earth Elements", "lat": 49.50, "lon": 92.50, "radius": 0.7, "mines": ["Mushgai Khudag", "Khalzan Buregtei", "Lugiin Gol"]},
            {"state": "Sukhbaatar", "district": "Eastern Mongolian VMS", "commod": "Zinc", "lat": 46.70, "lon": 113.25, "radius": 0.6, "mines": ["Tumurtin-Ovoo", "Ulaan", "Tsav"]}
        ]
    },
    "Japan": {
        "region": "AS",
        "belts": [
            {"state": "Gifu", "district": "Hida Metamorphic Complex", "commod": "Zinc", "lat": 36.40, "lon": 137.30, "radius": 0.5, "mines": ["Kamioka Mozumi", "Kamioka Tochibora", "Hiragane"]},
            {"state": "Ehime", "district": "Sanbagawa Metamorphic Belt", "commod": "Copper", "lat": 33.90, "lon": 133.30, "radius": 0.5, "mines": ["Besshi", "Motoyama", "Shirataki"]},
            {"state": "Akita", "district": "Hokuroku Kuroko VMS Belt", "commod": "Zinc", "lat": 40.30, "lon": 140.60, "radius": 0.5, "mines": ["Kosaka", "Hanaoka", "Matsumine", "Furutobe"]},
            {"state": "Tokyo", "district": "Minamitorishima Deep Sea REE", "commod": "Rare Earth Elements", "lat": 24.30, "lon": 153.95, "radius": 0.6, "mines": ["Minamitorishima Abyssal", "Marcus Trench Mud", "Ogasawara Basin"]},
            {"state": "Kagoshima", "district": "Southern Kyushu Volcanic Arc", "commod": "Lithium", "lat": 31.90, "lon": 130.60, "radius": 0.5, "mines": ["Hishikari Brine", "Yamagawa", "Kirishima"]}
        ]
    },
    "Saudi Arabia": {
        "region": "AS",
        "belts": [
            {"state": "Medina", "district": "Arabian Shield VMS Belt", "commod": "Copper", "lat": 23.85, "lon": 40.85, "radius": 0.6, "mines": ["Jabal Sayid", "Mahd Ad Dhahab", "Bulghah"]},
            {"state": "Riyadh", "district": "Al Amar Suture Belt", "commod": "Zinc", "lat": 23.75, "lon": 45.05, "radius": 0.5, "mines": ["Al Amar", "Khnaiguiyah", "Umm Ash Shalahib"]},
            {"state": "Tabuk", "district": "Midyan Alkaline Complex", "commod": "Rare Earth Elements", "lat": 28.00, "lon": 35.50, "radius": 0.6, "mines": ["Ghurayyah Ta-REE", "Jabal Tawlah", "Midyan"]},
            {"state": "Hail", "district": "Central Arabian Bauxite Belt", "commod": "Aluminum", "lat": 27.50, "lon": 43.50, "radius": 0.7, "mines": ["Al Ba'itha", "Zubirah", "Qassim"]},
            {"state": "Makkah", "district": "Wadi Fatima Titanomagnetite", "commod": "Titanium", "lat": 21.50, "lon": 39.80, "radius": 0.5, "mines": ["Wadi Fatima", "Jabal Idsas", "Al Wajh"]}
        ]
    },
    "Malaysia": {
        "region": "AS",
        "belts": [
            {"state": "Pahang", "district": "Kuantan Lateritic Bauxite", "commod": "Aluminum", "lat": 3.85, "lon": 103.30, "radius": 0.5, "mines": ["Kuantan Bauxite", "Bukit Goh", "Beserah"]},
            {"state": "Johor", "district": "Pengerang Coastal Bauxite", "commod": "Aluminum", "lat": 1.35, "lon": 104.15, "radius": 0.4, "mines": ["Pengerang", "Kota Tinggi", "Teluk Ramunia"]},
            {"state": "Perak", "district": "Kinta Valley Pegmatite/Placer", "commod": "Rare Earth Elements", "lat": 4.50, "lon": 101.10, "radius": 0.5, "mines": ["Kinta Monazite", "Kroh", "Grik Ion Clay"]},
            {"state": "Sabah", "district": "Central Sabah Ophiolite Belt", "commod": "Nickel", "lat": 5.80, "lon": 116.85, "radius": 0.6, "mines": ["Mamut Copper", "Ranau Nickel", "Telupid"]},
            {"state": "Terengganu", "district": "Eastern Tin-Titanium Belt", "commod": "Titanium", "lat": 5.30, "lon": 103.00, "radius": 0.5, "mines": ["Kemaman Ilmenite", "Dungun", "Marang"]}
        ]
    },

    # ---------------- Europe (12) ----------------
    "Russia": {
        "region": "EU",
        "belts": [
            {"state": "Krasnoyarsk Krai", "district": "Norilsk-Talnakh Intrusive Complex", "commod": "Nickel", "lat": 69.35, "lon": 88.20, "radius": 0.6, "mines": ["Oktyabrsky", "Komsomolsky", "Taimyrsky", "Mayak", "Skalisty"]},
            {"state": "Murmansk Oblast", "district": "Kola Peninsula Mafic Complex", "commod": "Platinum", "lat": 67.55, "lon": 34.00, "radius": 0.8, "mines": ["Monchegorsk", "Fedorova Tundra", "Severny", "Zapolyarny"]},
            {"state": "Murmansk Oblast", "district": "Lovozero Carbonatite Complex", "commod": "Rare Earth Elements", "lat": 67.85, "lon": 34.65, "radius": 0.5, "mines": ["Karnasurt", "Umbozero", "Khibiny Apatite-REE"]},
            {"state": "Zabaykalsky Krai", "district": "Udokan Sandstone Copper Basin", "commod": "Copper", "lat": 56.55, "lon": 118.60, "radius": 0.6, "mines": ["Udokan", "Chitkanda", "Unkur"]},
            {"state": "Chelyabinsk Oblast", "district": "Southern Urals VMS Belt", "commod": "Copper", "lat": 53.50, "lon": 59.20, "radius": 0.7, "mines": ["Gaiskoye", "Uchalinskoe", "Sibay", "Alexandrinskoye"]},
            {"state": "Sverdlovsk Oblast", "district": "Kachkanar Titanomagnetite", "commod": "Vanadium", "lat": 58.70, "lon": 59.50, "radius": 0.5, "mines": ["Kachkanar", "Gusevogorskoye", "Pervouralskoye"]},
            {"state": "Irkutsk Oblast", "district": "Zashikhinskoe Rare Metal Complex", "commod": "Lithium", "lat": 54.20, "lon": 99.50, "radius": 0.6, "mines": ["Zavitinskoye", "Golsovoye", "Tastyg"]}
        ]
    },
    "Finland": {
        "region": "EU",
        "belts": [
            {"state": "Lapland", "district": "Central Lapland Greenstone Belt", "commod": "Nickel", "lat": 67.70, "lon": 26.85, "radius": 0.6, "mines": ["Kevitsa", "Sakatti", "Pahtavaara"]},
            {"state": "Kainuu", "district": "Talvivaara Black Shale Basin", "commod": "Cobalt", "lat": 63.98, "lon": 28.00, "radius": 0.5, "mines": ["Terrafame Talvivaara", "Kuusilampi", "Kolmisoppi"]},
            {"state": "Central Ostrobothnia", "district": "Kaustinen Pegmatite Province", "commod": "Lithium", "lat": 63.55, "lon": 23.75, "radius": 0.4, "mines": ["Keliber Syväjärvi", "Rapasaari", "Länttä", "Outovesi"]},
            {"state": "North Karelia", "district": "Outokumpu Copper-Cobalt Belt", "commod": "Copper", "lat": 62.75, "lon": 29.00, "radius": 0.5, "mines": ["Keretti", "Vuonos", "Kylylahti"]},
            {"state": "Lapland", "district": "Suhanko Layered Intrusion", "commod": "Platinum", "lat": 66.15, "lon": 25.40, "radius": 0.5, "mines": ["Suhanko Konttijärvi", "Ahmavaara", "Portimo"]}
        ]
    },
    "Sweden": {
        "region": "EU",
        "belts": [
            {"state": "Norrbotten", "district": "Jokkmokk-Kiruna Ore Belt", "commod": "Copper", "lat": 67.15, "lon": 20.65, "radius": 0.7, "mines": ["Aitik", "Liikavaara", "Nautanen", "Viscaria"]},
            {"state": "Orebro", "district": "Bergslagen Mineral Province", "commod": "Zinc", "lat": 58.80, "lon": 15.10, "radius": 0.6, "mines": ["Zinkgruvan", "Garpenberg", "Lovisa", "Ammeberg"]},
            {"state": "Jonkoping", "district": "Norra Karr Peralkaline Complex", "commod": "Rare Earth Elements", "lat": 58.10, "lon": 14.55, "radius": 0.4, "mines": ["Norra Karr", "Grenna", "Almunge"]},
            {"state": "Vasterbotten", "district": "Skellefte VMS Belt", "commod": "Cobalt", "lat": 65.00, "lon": 20.00, "radius": 0.6, "mines": ["Kristineberg", "Renstrom", "Petiknas"]},
            {"state": "Vasternorrland", "district": "Jamtland Vanadium Shales", "commod": "Vanadium", "lat": 63.00, "lon": 15.00, "radius": 0.6, "mines": ["Haggan", "Viken", "Tasarmin"]}
        ]
    },
    "Norway": {
        "region": "EU",
        "belts": [
            {"state": "Vestland", "district": "Engebo Rutile-Garnet Complex", "commod": "Titanium", "lat": 61.50, "lon": 5.45, "radius": 0.4, "mines": ["Engebo Mountain", "Naustdal", "Fordefjord"]},
            {"state": "Rogaland", "district": "Egersund Anorthosite Complex", "commod": "Titanium", "lat": 58.35, "lon": 6.40, "radius": 0.5, "mines": ["Tellnes Titania", "Rodsand", "Koldal"]},
            {"state": "Nordland", "district": "Rana Metamorphic Belt", "commod": "Graphite", "lat": 66.30, "lon": 14.15, "radius": 0.6, "mines": ["Skaland Graphite", "Jennestad", "Holandsvika"]},
            {"state": "Telemark", "district": "Fen Carbonatite Complex", "commod": "Rare Earth Elements", "lat": 59.28, "lon": 9.30, "radius": 0.4, "mines": ["Fen Complex", "Rodberg", "Gruvefjellet"]},
            {"state": "Trondelag", "district": "Roros-Lokken VMS Belt", "commod": "Copper", "lat": 63.15, "lon": 9.70, "radius": 0.5, "mines": ["Lokken Verk", "Roros", "Killingdal"]}
        ]
    },
    "Germany": {
        "region": "EU",
        "belts": [
            {"state": "Saxony", "district": "Erzgebirge Pegmatite/Greisen", "commod": "Lithium", "lat": 50.73, "lon": 13.75, "radius": 0.4, "mines": ["Zinnwald Lithium", "Sadisdorf", "Falkenhain"]},
            {"state": "Baden-Wurttemberg", "district": "Upper Rhine Graben Geothermal", "commod": "Lithium", "lat": 49.00, "lon": 8.40, "radius": 0.6, "mines": ["Vulcan Insheim", "Landau Geothermal", "Bruchsal"]},
            {"state": "Bavaria", "district": "Passau Kropfmühl Belt", "commod": "Graphite", "lat": 48.60, "lon": 13.65, "radius": 0.4, "mines": ["Kropfmuhl", "Pfaffenreuth", "Hauzenberg"]},
            {"state": "Saxony-Anhalt", "district": "Mansfeld Kupferschiefer Basin", "commod": "Copper", "lat": 51.55, "lon": 11.55, "radius": 0.5, "mines": ["Mansfeld", "Sangerhausen", "Eisleben"]},
            {"state": "Lower Saxony", "district": "Rammelsberg SEDEX District", "commod": "Zinc", "lat": 51.88, "lon": 10.42, "radius": 0.4, "mines": ["Rammelsberg", "Bad Grund", "Meggen"]}
        ]
    },
    "Poland": {
        "region": "EU",
        "belts": [
            {"state": "Lower Silesia", "district": "Foresudetic Kupferschiefer Basin", "commod": "Copper", "lat": 51.50, "lon": 16.15, "radius": 0.6, "mines": ["Rudna KGHM", "Polkowice-Sieroszowice", "Lubin", "Glogow Gleboki"]},
            {"state": "Lower Silesia", "district": "Kupferschiefer Stratiform Cobalt", "commod": "Cobalt", "lat": 51.40, "lon": 16.05, "radius": 0.5, "mines": ["Konrad", "Lena", "Nowy Kosciol"]},
            {"state": "Silesian", "district": "Olkusz-Pomorzany MVT Basin", "commod": "Zinc", "lat": 50.30, "lon": 19.50, "radius": 0.5, "mines": ["Pomorzany", "Olkusz", "Trzebionka", "Boleslaw"]},
            {"state": "Podlaskie", "district": "Suwalki Anorthosite Massif", "commod": "Vanadium", "lat": 54.10, "lon": 22.90, "radius": 0.5, "mines": ["Krzemianka", "Udryn", "Jeleniewo"]},
            {"state": "Lower Silesia", "district": "Strzeblow Metamorphic District", "commod": "Nickel", "lat": 50.90, "lon": 16.70, "radius": 0.5, "mines": ["Szklary Laterite", "Gromnik", "Jordanow"]}
        ]
    },
    "Spain": {
        "region": "EU",
        "belts": [
            {"state": "Andalusia", "district": "Iberian Pyrite Belt", "commod": "Copper", "lat": 37.60, "lon": -6.60, "radius": 0.6, "mines": ["Rio Tinto", "Aguas Tenidas", "Cobre Las Cruces", "Magdalena"]},
            {"state": "Andalusia", "district": "Iberian VMS Polymetallic", "commod": "Zinc", "lat": 37.75, "lon": -6.95, "radius": 0.5, "mines": ["Sotiel", "Tharsis", "Aznalcollar", "Los Frailes"]},
            {"state": "Extremadura", "district": "Valdeflorez Pegmatite Field", "commod": "Lithium", "lat": 39.45, "lon": -6.35, "radius": 0.4, "mines": ["San Jose Valdeflorez", "Las Navas", "Canoro"]},
            {"state": "Galicia", "district": "Lalin Pegmatite Province", "commod": "Lithium", "lat": 42.65, "lon": -8.15, "radius": 0.5, "mines": ["Doade", "Gondomar", "Tres Cerros"]},
            {"state": "Galicia", "district": "Aguablanca Mafic Intrusion", "commod": "Nickel", "lat": 38.00, "lon": -6.20, "radius": 0.4, "mines": ["Aguablanca", "Monchi", "Ossa Morena"]}
        ]
    },
    "Portugal": {
        "region": "EU",
        "belts": [
            {"state": "Vila Real", "district": "Barroso Pegmatite Field", "commod": "Lithium", "lat": 41.68, "lon": -7.72, "radius": 0.4, "mines": ["Mina do Barroso", "Grandao", "Reservatorio", "NOA"]},
            {"state": "Guarda", "district": "Montalegre-Sepeda Corridor", "commod": "Lithium", "lat": 40.80, "lon": -7.10, "radius": 0.5, "mines": ["Romano", "Sepeda", "Alvarroes"]},
            {"state": "Beja", "district": "Iberian Pyrite Belt South", "commod": "Copper", "lat": 37.60, "lon": -7.98, "radius": 0.5, "mines": ["Neves-Corvo", "Aljustrel", "Sao Domingos", "Caveira"]},
            {"state": "Castelo Branco", "district": "Panasqueira Greisen Belt", "commod": "Zinc", "lat": 40.15, "lon": -7.75, "radius": 0.4, "mines": ["Panasqueira Zn", "Barroca", "Boidobra"]},
            {"state": "Portalegre", "district": "Ossa Morena Cobalt Belt", "commod": "Cobalt", "lat": 39.30, "lon": -7.45, "radius": 0.5, "mines": ["Portalegre Co", "Arronches", "Monforte"]}
        ]
    },
    "France": {
        "region": "EU",
        "belts": [
            {"state": "Auvergne-Rhone-Alpes", "district": "Echassieres Beauvoir Granite", "commod": "Lithium", "lat": 46.18, "lon": 2.95, "radius": 0.4, "mines": ["Beauvoir EMILI", "Colettes", "Bessay"]},
            {"state": "Grand Est", "district": "Alsace Geothermal Lithium Basin", "commod": "Lithium", "lat": 48.75, "lon": 7.85, "radius": 0.5, "mines": ["Rittershoffen", "Soultz-sous-Forets", "Vendenheim"]},
            {"state": "Brittany", "district": "Armorican VMS Belt", "commod": "Zinc", "lat": 48.20, "lon": -3.50, "radius": 0.6, "mines": ["Porte-aux-Moines", "Bodennec", "Rouzic"]},
            {"state": "Occitanie", "district": "Cevennes Polymetallic District", "commod": "Copper", "lat": 44.15, "lon": 3.80, "radius": 0.5, "mines": ["Sentein", "Saint-Felix", "Peyrebrune"]},
            {"state": "Provence", "district": "Alpilles Bauxite Basin", "commod": "Aluminum", "lat": 43.70, "lon": 4.85, "radius": 0.5, "mines": ["Les Baux-de-Provence", "Brignoles", "Cabasse"]}
        ]
    },
    "United Kingdom": {
        "region": "EU",
        "belts": [
            {"state": "England", "district": "Cornwall Granite Pegmatite & Brine", "commod": "Lithium", "lat": 50.25, "lon": -5.20, "radius": 0.4, "mines": ["Cornish Lithium Trelavour", "United Downs Brine", "South Crofty"]},
            {"state": "England", "district": "Cornwall Historic Mining Belt", "commod": "Copper", "lat": 50.20, "lon": -5.30, "radius": 0.5, "mines": ["Dolcoath", "Geevor", "Levant", "Pendarves"]},
            {"state": "England", "district": "Northern Pennines Orefield", "commod": "Zinc", "lat": 54.75, "lon": -2.25, "radius": 0.5, "mines": ["Nenthead", "Allenheads", "Groverake"]},
            {"state": "Scotland", "district": "Grampian VMS Belt", "commod": "Nickel", "lat": 57.10, "lon": -3.50, "radius": 0.6, "mines": ["Arthrath", "Loch Borralan REE", "Tyndrum"]},
            {"state": "Wales", "district": "Parys Mountain VMS District", "commod": "Copper", "lat": 53.38, "lon": -4.35, "radius": 0.3, "mines": ["Parys Mountain", "Mona", "Ogofau"]}
        ]
    },
    "Serbia": {
        "region": "EU",
        "belts": [
            {"state": "Macva", "district": "Jadar Borate-Lithium Basin", "commod": "Lithium", "lat": 44.55, "lon": 19.30, "radius": 0.4, "mines": ["Jadar Project", "Nedeljice", "Brezjak"]},
            {"state": "Bor", "district": "Timok Magmatic Complex", "commod": "Copper", "lat": 44.05, "lon": 22.10, "radius": 0.5, "mines": ["Cukaru Peki", "Bor Open Pit", "Veliki Krivelj", "Majdanpek"]},
            {"state": "Raska", "district": "Kopaonik Polymetallic Belt", "commod": "Zinc", "lat": 43.25, "lon": 20.80, "radius": 0.5, "mines": ["Suva Ruda", "Kizevak", "Sastavci", "Grot"]},
            {"state": "Sumadija", "district": "Rudnik Mining District", "commod": "Zinc", "lat": 44.15, "lon": 20.50, "radius": 0.4, "mines": ["Rudnik", "Beli Kamen", "Crnac"]},
            {"state": "Zlatibor", "district": "Mokra Gora Ophiolite", "commod": "Nickel", "lat": 43.80, "lon": 19.50, "radius": 0.5, "mines": ["Mokra Gora", "Lipovac", "Vardište"]}
        ]
    },
    "Austria": {
        "region": "EU",
        "belts": [
            {"state": "Carinthia", "district": "Koralpe Pegmatite District", "commod": "Lithium", "lat": 46.85, "lon": 14.95, "radius": 0.4, "mines": ["Wolfsberg Lithium", "Brandruecken", "Weinebene"]},
            {"state": "Salzburg", "district": "Felbertal Scheelite-Vanadium", "commod": "Vanadium", "lat": 47.15, "lon": 12.50, "radius": 0.4, "mines": ["Mittersill Felbertal", "Habachtal", "Bramberg"]},
            {"state": "Tyrol", "district": "Brixlegg Fahlore District", "commod": "Copper", "lat": 47.43, "lon": 11.88, "radius": 0.4, "mines": ["Brixlegg", "Schwaz", "Grosskogel"]},
            {"state": "Carinthia", "district": "Bleiberg-Kreuth SEDEX/MVT", "commod": "Zinc", "lat": 46.62, "lon": 13.68, "radius": 0.4, "mines": ["Bleiberg", "Kreuth", "Rudolf Schacht"]},
            {"state": "Styria", "district": "Erzberg Carbonate Massif", "commod": "Manganese", "lat": 47.53, "lon": 14.90, "radius": 0.4, "mines": ["Steirischer Erzberg", "Radmer", "Vordernberg"]}
        ]
    },

    # ---------------- Oceania (4) ----------------
    "Australia": {
        "region": "OC",
        "belts": [
            {"state": "Western Australia", "district": "Pilbara Pegmatite Province", "commod": "Lithium", "lat": -21.05, "lon": 118.90, "radius": 0.7, "mines": ["Pilgangoora", "Wodgina", "Tabba Tabba", "Marble Bar"]},
            {"state": "Western Australia", "district": "South West Yilgarn Craton", "commod": "Lithium", "lat": -33.85, "lon": 116.05, "radius": 0.6, "mines": ["Greenbushes", "Mt Cattlin", "Bald Hill", "Bulgaria Pegmatite"]},
            {"state": "Western Australia", "district": "Eastern Goldfields / Yilgarn", "commod": "Nickel", "lat": -31.20, "lon": 121.65, "radius": 0.8, "mines": ["Kambalda", "Mt Keith", "Leinster", "Murrin Murrin HPAL", "Nova-Bollinger"]},
            {"state": "South Australia", "district": "Olympic Copper-Gold Province", "commod": "Copper", "lat": -30.45, "lon": 136.88, "radius": 0.7, "mines": ["Olympic Dam", "Prominent Hill", "Carrapateena", "Oak Dam"]},
            {"state": "Western Australia", "district": "Northern Yilgarn REE Carbonatite", "commod": "Rare Earth Elements", "lat": -28.80, "lon": 122.30, "radius": 0.5, "mines": ["Mount Weld", "Cummins Range", "Yangibana", "Browns Range"]},
            {"state": "Northern Territory", "district": "Groote Eylandt Basin", "commod": "Manganese", "lat": -13.95, "lon": 136.45, "radius": 0.4, "mines": ["Groote Eylandt GEMCO", "Bootu Creek", "Woodie Woodie"]}
        ]
    },
    "Papua New Guinea": {
        "region": "OC",
        "belts": [
            {"state": "Madang", "district": "Ramu Ultramafic Complex", "commod": "Nickel", "lat": -5.60, "lon": 145.25, "radius": 0.5, "mines": ["Ramu HPAL", "Kurumbukari", "Basamuk"]},
            {"state": "Western Province", "district": "Papuan Fold Belt Porphyry", "commod": "Copper", "lat": -5.05, "lon": 141.15, "radius": 0.5, "mines": ["Ok Tedi", "Mt Fubilan", "Wafi-Golpu"]},
            {"state": "Bougainville", "district": "Crown Prince Range", "commod": "Copper", "lat": -6.32, "lon": 155.50, "radius": 0.4, "mines": ["Panguna", "Kupei", "Moroni"]},
            {"state": "New Ireland", "district": "Tabar-Lihir Volcanic Arc", "commod": "Cobalt", "lat": -3.12, "lon": 152.62, "radius": 0.5, "mines": ["Lihir Luise", "Simberi", "Niolam"]}
        ]
    },
    "New Caledonia": {
        "region": "OC",
        "belts": [
            {"state": "North Province", "district": "Massif du Nord Ultramafic Belt", "commod": "Nickel", "lat": -20.95, "lon": 164.80, "radius": 0.6, "mines": ["Koniambo", "Tiebaghi", "Poum", "Kaala"]},
            {"state": "South Province", "district": "Grand Massif du Sud", "commod": "Cobalt", "lat": -22.25, "lon": 166.90, "radius": 0.6, "mines": ["Goro HPAL", "Prony", "Kouaoua", "Thio"]},
            {"state": "North Province", "district": "Nakety-Canala Corridor", "commod": "Nickel", "lat": -21.50, "lon": 165.95, "radius": 0.5, "mines": ["Canala", "Nakety", "Boakaine"]},
            {"state": "South Province", "district": "Plateau de Poro", "commod": "Cobalt", "lat": -21.30, "lon": 165.75, "radius": 0.5, "mines": ["Poro", "Kouaoua South", "Vulcan"]}
        ]
    },
    "New Zealand": {
        "region": "OC",
        "belts": [
            {"state": "Waikato", "district": "Waikato Titanomagnetite Sands", "commod": "Titanium", "lat": -37.40, "lon": 174.70, "radius": 0.6, "mines": ["Waikato North Head", "Taharoa Ironsands", "Maioro"]},
            {"state": "Taranaki", "district": "Taranaki Offshore-Onshore Sands", "commod": "Vanadium", "lat": -39.05, "lon": 174.00, "radius": 0.6, "mines": ["Taranaki VTM", "Patea", "Wanganui Sands"]},
            {"state": "Otago", "district": "Otago Metamorphic Schist Belt", "commod": "Copper", "lat": -45.40, "lon": 170.40, "radius": 0.6, "mines": ["Macraes", "Moke Creek", "Waitahuna"]},
            {"state": "West Coast", "district": "Greenland Group Polymetallic", "commod": "Zinc", "lat": -42.45, "lon": 171.20, "radius": 0.5, "mines": ["Blackball", "Reefton Zn", "Kumara"]}
        ]
    }
}

def load_mrds_ev_records():
    """Extract and clean authentic EV records from USGS MRDS files."""
    print("Loading USGS MRDS records from rdbms...")
    
    # 1. Commodities
    com_path = os.path.join(BASE_RDBMS, "Commodity.txt")
    com_df = pd.read_csv(com_path, sep="\t", usecols=["dep_id", "commod"], low_memory=False)
    com_df["commod_clean"] = com_df["commod"].map(COMMOD_MAP)
    com_df = com_df.dropna(subset=["commod_clean"]).drop_duplicates(subset=["dep_id"])
    
    # 2. Locations
    loc_path = os.path.join(BASE_RDBMS, "Location.txt")
    loc_df = pd.read_csv(loc_path, sep="\t", usecols=["dep_id", "country", "state_prov", "county", "region"], low_memory=False)
    # Fix 'america' artifact and normalize
    loc_df["country"] = loc_df["country"].replace({
        "america": "United States",
        "USA": "United States",
        "Congo (Kinshasa)": "DR Congo",
        "Congo (Brazzaville)": "DR Congo"
    })
    
    # 3. Coords
    coord_path = os.path.join(BASE_RDBMS, "Coords.txt")
    coord_df = pd.read_csv(coord_path, sep="\t", usecols=["dep_id", "wgs84_lat", "wgs84_lon"], low_memory=False).drop_duplicates(subset=["dep_id"])
    
    # 4. Names
    name_path = os.path.join(BASE_RDBMS, "Names.txt")
    name_df = pd.read_csv(name_path, sep="\t", usecols=["dep_id", "name"], low_memory=False).drop_duplicates(subset=["dep_id"])
    
    # 5. Districts
    dist_path = os.path.join(BASE_RDBMS, "Districts.txt")
    dist_df = pd.read_csv(dist_path, sep="\t", usecols=["dep_id", "district"], low_memory=False).drop_duplicates(subset=["dep_id"])
    
    # Merge
    merged = com_df.merge(loc_df, on="dep_id").merge(coord_df, on="dep_id").merge(name_df, on="dep_id", how="left").merge(dist_df, on="dep_id", how="left")
    
    merged["commod"] = merged["commod_clean"]
    merged = merged.dropna(subset=["wgs84_lat", "wgs84_lon", "country"])
    merged = merged[merged["country"].isin(COUNTRY_PROFILES.keys())]
    
    print(f"Extracted {len(merged)} valid authentic records matching target countries and EV commodities.")
    return merged

def generate_balanced_dataset():
    """Build the consolidated 100,000-row globally balanced dataset."""
    mrds_df = load_mrds_ev_records()
    
    all_rows = []
    current_dep_id = 90000000
    
    print("\nSynthesizing and balancing 2,000 rows for each of the 50 countries...")
    
    for country, profile in COUNTRY_PROFILES.items():
        region = profile["region"]
        belts = profile["belts"]
        
        # Take real MRDS records if available for this country
        country_mrds = mrds_df[mrds_df["country"] == country]
        num_mrds = len(country_mrds)
        
        # Cap real records at 1,000 to allow balanced representation of core belts and exploration targets
        take_mrds = min(num_mrds, 1000)
        
        country_rows = []
        if take_mrds > 0:
            sample_mrds = country_mrds.sample(n=take_mrds, random_state=42)
            for _, r in sample_mrds.iterrows():
                current_dep_id += 1
                # Realistic grade distribution: 50% A, 50% B
                grade = "A" if random.random() < 0.52 else "B"
                country_rows.append({
                    "dep_id": current_dep_id,
                    "commod": str(r["commod"]),
                    "final_grade": grade,
                    "wgs84_lat": round(float(r["wgs84_lat"]), 5),
                    "wgs84_lon": round(float(r["wgs84_lon"]), 5),
                    "region": region,
                    "country": country,
                    "state_prov": str(r["state_prov"]) if pd.notnull(r["state_prov"]) and str(r["state_prov"]) != "NULL" else belts[0]["state"],
                    "county": str(r["county"]) if pd.notnull(r["county"]) and str(r["county"]) != "NULL" else "NULL",
                    "name": str(r["name"]) if pd.notnull(r["name"]) and str(r["name"]) != "NULL" else f"{country} Mineral Deposit",
                    "district": str(r["district"]) if pd.notnull(r["district"]) and str(r["district"]) != "NULL" else belts[0]["district"]
                })
        
        # Synthesize remaining rows up to exactly 2,000 using authentic metallogenic belts
        needed = 2000 - len(country_rows)
        for i in range(needed):
            current_dep_id += 1
            belt = random.choice(belts)
            
            # Geostatistical dispersion around real belt centroid
            lat_disp = np.random.normal(0, belt["radius"] * 0.45)
            lon_disp = np.random.normal(0, belt["radius"] * 0.45)
            lat = round(belt["lat"] + lat_disp, 5)
            lon = round(belt["lon"] + lon_disp, 5)
            
            # Select mine/deposit naming pattern
            mine_base = random.choice(belt["mines"])
            suffixes = ["East", "West", "North", "South", "Deep", "Pit No. 1", "Deposit II", "Extension", "Project", "Consolidated", "Splay", "Prospect"]
            deposit_name = f"{mine_base} {random.choice(suffixes)}" if random.random() < 0.65 else mine_base
            
            # Exploration grade assignment
            grade = "A" if random.random() < 0.50 else "B"
            
            country_rows.append({
                "dep_id": current_dep_id,
                "commod": belt["commod"],
                "final_grade": grade,
                "wgs84_lat": lat,
                "wgs84_lon": lon,
                "region": region,
                "country": country,
                "state_prov": belt["state"],
                "county": "NULL",
                "name": deposit_name,
                "district": belt["district"]
            })
            
        all_rows.extend(country_rows)
        print(f" -> {country} [{region}]: {len(country_rows)} records (MRDS: {take_mrds}, Metallogenic Belt: {needed})")
        
    df_final = pd.DataFrame(all_rows)
    
    # Shuffle for uniform distribution across rows
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\nDataset generation complete!")
    print(f"Total Rows: {len(df_final)}")
    print(f"Unique Countries: {df_final['country'].nunique()}")
    print("Country distribution (min, max):", df_final['country'].value_counts().min(), df_final['country'].value_counts().max())
    print("\nCommodity breakdown:\n", df_final['commod'].value_counts())
    print("\nGrade breakdown:\n", df_final['final_grade'].value_counts())
    
    # Save to CSV
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSuccessfully wrote new dataset to: {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_balanced_dataset()
