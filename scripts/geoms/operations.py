import geopandas as gpd
from shapely import from_wkt
import pandas as pd
import overpass
from shapely.geometry import shape

def clean_comm_areas(in_file, out_file) -> gpd.GeoDataFrame:
    df = pd.read_csv(in_file)
    df['geometry'] = df['the_geom'].apply(from_wkt)
    assert df['AREA_NUMBE'].nunique() == len(df)
    assert all(df['AREA_NUM_1'] == df['AREA_NUMBE'])
    df['id'] = df['AREA_NUMBE']
    df = df.filter(['id','geometry','COMMUNITY'])
    df = df.rename(columns={'COMMUNITY':'community_name'})
    df = gpd.GeoDataFrame(df, geometry='geometry')
    df.to_parquet(out_file)
    return df

street_type_codes = {
    "AVE": "AVENUE",
    "BLVD": "BOULEVARD",
    "CRES": "CRESCENT COURT",
    "CT": "COURT",
    "DR": "DRIVE",
    "ER": "ENTRANCE RAMP",
    "EXPY": "EXPRESSWAY",
    "HWY": "HIGHWAY",
    "LN": "LANE",
    "PKWY": "PARK WAY",
    "PL": "PLACE",
    "ROW": "ROW",
    "SQ": "SQUARE",
    "SR": "SERVICE ROAD",
    "ST": "STREET",
    "TER": "TERRACE",
    "TOLL": "TOLL WAY",
    "VIA": "WAY",
    "WAY": "EXIT RAMP"
}

suf_dir_codes = {
    "N": "NORTH",
    "S": "SOUTH",
    "E": "EAST",
    "W": "WEST",
    "OP": "OVERPASS",
    "IB": "INBOUND",
    "OB": "OUTBOUND",
    "NB": "NORTHBOUND",
    "SB": "SOUTHBOUND",
    "EB": "EASTBOUND",
    "WB": "WESTBOUND"
}

sides = ['West Side', 'South Side', 'North Side', 'Northwest Side', 'Southwest Side']

def _full_name(df:pd.DataFrame, pre:str, nam:str, typ:str, suf:str) -> pd.Series:
    result = df[pre].fillna('') + " " + df[nam].fillna('') + " " + df[typ].fillna('') + " " + df[suf].fillna('')
    return result.str.strip()

def clean_street_segs(in_file, out_file) -> gpd.GeoDataFrame:
    df = gpd.read_file(in_file)
    cols = ['PRE_DIR','STREET_NAM','STREET_TYP','SUF_DIR','L_F_ADD','L_T_ADD','R_F_ADD','R_T_ADD','F_CROSS','T_CROSS','LENGTH','L_ZIP','R_ZIP']
    cols = [x.lower() for x in cols]
    assert all([c in df.columns for c in cols])
    df = df.filter(cols + ['geometry'])

    df['from_num'] = pd.to_numeric(df[['l_f_add','r_f_add']].min(axis=1), 'coerce')
    df['to_num'] = pd.to_numeric(df[['l_t_add','r_t_add']].max(axis=1), 'coerce')
    df = df.drop(['l_f_add','r_f_add','l_t_add','r_t_add'], axis=1)

    df['street_typ_exp'] = df['street_typ'].replace(street_type_codes)
    df['suf_dir_exp'] = df['suf_dir'].replace(suf_dir_codes)
    df['pre_dir_exp'] = df['pre_dir'].replace(suf_dir_codes)

    df['street_typ_exp'] = df['street_typ_exp'].str.title()
    df['suf_dir_exp'] = df['suf_dir_exp'].str.title()
    df['pre_dir_exp'] = df['pre_dir_exp'].str.title()
    df['street_nam'] = df['street_nam'].str.title()
    df['street_typ'] = df['street_typ'].str.title()

    df['full_name1'] = df.pipe(_full_name, 'pre_dir','street_nam','street_typ','suf_dir')
    df['full_name2'] = df.pipe(_full_name, 'pre_dir_exp','street_nam','street_typ','suf_dir_exp')
    df['full_name3'] = df.pipe(_full_name, 'pre_dir','street_nam','street_typ_exp','suf_dir')
    df['full_name4'] = df.pipe(_full_name, 'pre_dir_exp','street_nam','street_typ_exp','suf_dir_exp')
    df.to_parquet(out_file)
    return df

def clean_street_names(in_file, out_file) -> pd.DataFrame: 
    df = pd.read_csv(in_file)
    df.columns = [c.strip() for c in df.columns]
    df['street_partial'] = df['Street'] + " " + df['Suffix']
    df.to_csv(out_file, index=False)
    return df

def create_blocks(streets_in_file, out_file) -> gpd.GeoDataFrame:
    street_segs = gpd.read_parquet(streets_in_file)
    street_blocks = []
    for row in street_segs.itertuples():
        for block in filter(lambda x: x % 100 == 0, range(row.from_num, row.to_num + 1)):
            street_blocks.append({
                'geometry': row.geometry,
                'block_name1': str(block) + " block of " + str(row.full_name1),
                'block_name2': str(block) + " block of " + str(row.full_name2),
                'block_name3': str(block) + " block of " + str(row.full_name3),
                'block_name4': str(block) + " block of " + str(row.full_name4),
            })
    df = gpd.GeoDataFrame(street_blocks, geometry='geometry')
    df.to_parquet(out_file)
    return df

def create_intersections(segs_in_path, out_file) -> pd.DataFrame:
    segments = gpd.read_parquet(segs_in_path)
    segment_names = segments['pre_dir'] + " " + segments['street_nam']
    segment_names_full = segments['pre_dir'] + " " + segments['street_nam'] + " " + segments['street_typ']
    cross_streets_to = segments['t_cross'].str.replace('|',' ', regex=False).str.lstrip('1234567890').str.replace(r'\s+', ' ',regex=True).str.strip()
    cross_streets_from = segments['f_cross'].str.replace('|',' ', regex=False).str.lstrip('1234567890').str.replace(r'\s+', ' ',regex=True).str.strip()

    invalid_cross_from = cross_streets_from.str.count(' ') < 2
    invalid_cross_to = cross_streets_to.str.count(' ') < 2

    def enumerate_cross_streets(segments, crosses, mask):
        return pd.concat([
            (segments + " and " + crosses)[~mask],
            (segments + " & " + crosses)[~mask],
            (crosses + " and " + segments)[~mask],
            (crosses + " & " + segments)[~mask],
        ])
    intersections = pd.concat([
        enumerate_cross_streets(segment_names, cross_streets_to, invalid_cross_to),
        enumerate_cross_streets(segment_names, cross_streets_from, invalid_cross_from),
        enumerate_cross_streets(segment_names_full, cross_streets_to, invalid_cross_to),
        enumerate_cross_streets(segment_names_full, cross_streets_from, invalid_cross_from),
        ]).drop_duplicates().dropna().rename('intersection')
    intersections = pd.DataFrame(intersections)
    intersections.to_parquet(out_file)
    return intersections

def _query_overpass(query) -> gpd.GeoDataFrame:
    api = overpass.API()
    response = api.get(query)
    result = pd.DataFrame([
        dict(the_geom=feature['geometry'],
            street=feature['properties']['tags'].get('addr:street',None),
            housenumber=feature['properties']['tags'].get('addr:housenumber',None),
            name=feature['properties']['tags'].get('name',None))
        for feature in response['features']
    ])
    result['geometry'] = result['the_geom'].apply(shape)
    result['street'] = result['street'].str.strip()
    result['name'] = result['name'].str.strip()
    result = gpd.GeoDataFrame(result, geometry='geometry').drop(columns=['the_geom'])
    return result

hospital_query = """
(
  node["amenity"="hospital"](41.6445,-87.9401,42.0230,-87.5240);
  way["amenity"="hospital"](41.6445,-87.9401,42.0230,-87.5240);
  relation["amenity"="hospital"](41.6445,-87.9401,42.0230,-87.5240);
);
out center;
"""
landmark_query = """
(
  node["building"]["name"](41.6445,-87.9401,42.0230,-87.5240);
  way["building"]["name"](41.6445,-87.9401,42.0230,-87.5240);
  relation["building"]["name"](41.6445,-87.9401,42.0230,-87.5240);
);
out center;
"""
parks_query = """
(
  way["leisure"="park"](41.6445,-87.9401,42.0230,-87.5240);
  relation["leisure"="park"](41.6445,-87.9401,42.0230,-87.5240);
);
out geom;
"""

def get_hospitals(out_file) -> gpd.GeoDataFrame:
    df = _query_overpass(hospital_query)
    mask = df['name'].isna() | (df['name'].str.len() <= 2) | df['name'].duplicated(keep=False)
    df = df[~mask]
    df.to_parquet(out_file)
    return df

def get_landmarks(out_file) -> gpd.GeoDataFrame:
    df = _query_overpass(landmark_query)
    mask = df['name'].isna()
    df = df[~mask]
    df.to_parquet(out_file)
    return df

def get_parks(out_file) -> gpd.GeoDataFrame:
    df = _query_overpass(parks_query)
    mask = df['name'].isna()
    df = df[~mask]
    df.to_parquet(out_file)
    return df

def clean_neighborhoods(in_file, out_file) -> pd.Series:
    df = pd.read_csv(in_file)
    names = pd.concat([df['PRI_NEIGH'], df['SEC_NEIGH']])
    names = names.str.title().drop_duplicates().rename('name')
    names.to_csv(out_file, index=False)
    return names