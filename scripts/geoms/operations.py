import pygtrie
import re
import geopandas as gpd
from shapely import from_wkt
import pandas as pd
import overpass
from shapely.geometry import shape

from scripts.utils.logging import setup_logger

logger = setup_logger(__name__)

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
    "RD": "ROAD",
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
    df['dir_exp'] = df['Direction'].replace(suf_dir_codes)
    df['suf_exp'] = df['Suffix'].replace(street_type_codes)
    df['combined_name_1'] = df['Direction'] + " " + df['Street'] + " " + df['Suffix']
    df['combined_name_2'] = df['Direction'] + " " + df['Street'] + " " + df['suf_exp']
    df['combined_name_3'] = df['Direction'] + " " + df['Street']
    df['combined_name_4'] = df['dir_exp'] + " " + df['Street'] + " " + df['Suffix']
    df['combined_name_5'] = df['dir_exp'] + " " + df['Street'] + " " + df['suf_exp']
    df['combined_name_6'] = df['dir_exp'] + " " + df['Street']
    df['combined_name_7'] = df['Street'] + " " + df['Suffix']
    df['combined_name_8'] = df['Street'] + " " + df['suf_exp']
    # do NOT uncomment! [Street] by itself would be WAY to many false positives
    # e.g. [North] Ave, [Western] Ave, [Chicago] Ave, etc
    # df['combined_name_9'] = df['Street'] 
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

def map_blocks(street_block_file, comm_area_file, out_file):
    df = gpd.read_parquet(street_block_file)
    comm_areas = gpd.read_parquet(comm_area_file)
    
    df = df.sjoin(comm_areas[['community_name','geometry']],
                    how='inner',
                    predicate='intersects',
                    lsuffix='street',
                    rsuffix='comm').drop(columns=['index_comm','geometry'])
    
    df = (df.melt(id_vars=['community_name'],
                   value_vars=['block_name1', 'block_name2', 'block_name3', 'block_name4'],
                   var_name='exp',
                   value_name='block_name')
                .drop(columns=['exp'])
                .drop_duplicates())
    
    df[['block_name','community_name']].to_parquet(out_file)
    return df

def create_intersection_geoms(in_file, out_file):
    df = gpd.read_parquet(in_file)
    df = df.loc[(df.f_cross.str.count(r'\|') == 4) & (df.t_cross.str.count(r'\|') == 4)]
    df = df.assign(street_nam_upper = df.street_nam.str.upper(),
                   street_typ_upper = df.street_typ.str.upper(),
                   f_cross_nam = df.f_cross.str.upper().str.split('|', expand=False, regex=False).str[2],
                   f_cross_typ = df.f_cross.str.upper().str.split('|', expand=False, regex=False).str[3],
                   t_cross_nam = df.t_cross.str.upper().str.split('|', expand=False, regex=False).str[2],
                   t_cross_typ = df.t_cross.str.upper().str.split('|', expand=False, regex=False).str[3],
                   fullname1 = df.pre_dir.fillna('') + " " + df.street_nam.fillna('') + " " + df.street_typ.fillna(''),
                   fullname2 = df.pre_dir.fillna('') + " " + df.street_nam.fillna('') + " " + df.street_typ_exp.fillna(''),
                   fullname3 = df.pre_dir.fillna('') + " " + df.street_nam.fillna(''),
                   fullname4 = df.pre_dir_exp.fillna('') + " " + df.street_nam.fillna('') + " " + df.street_typ.fillna(''),
                   fullname5 = df.pre_dir_exp.fillna('') + " " + df.street_nam.fillna('') + " " + df.street_typ_exp.fillna(''),
                   fullname6 = df.pre_dir_exp.fillna('') + " " + df.street_nam.fillna(''),
                   fullname7 = df.street_nam.fillna('') + " " + df.street_typ.fillna(''),
                   fullname8 = df.street_nam.fillna('') + " " + df.street_typ_exp.fillna(''),
                   fullname9 = df.street_nam.fillna(''))
    qcuts = [.001,.002,.003,.004,.005,.006,.007,.008,.009,
            .01,.02,.03,.04,.05,.06,.07,.08,.09,
            .1,.2,.3,.4,.5,.6,.7,.8,.9]
    log_intervals = df.index.to_series().quantile(qcuts, 'lower')

    candidates_f = df.merge(df, how='inner', 
                            left_on=['street_nam_upper','street_typ_upper'],
                            right_on=['f_cross_nam','f_cross_typ'])

    candidates_t = df.merge(df, how='inner', 
                            left_on=['street_nam_upper','street_typ_upper'],
                            right_on=['t_cross_nam','t_cross_typ'])


    crosses = pygtrie.StringTrie()
    for seg in df.dropna(subset=['street_nam','geometry']).itertuples():
        if seg.Index in log_intervals.values:
            logger.debug("Processed {} ({} pct) segments".format(seg.Index, qcuts[log_intervals.searchsorted(seg.Index)]))
            logger.debug("crosses has {} elems".format(len(crosses)))
        matchall = pd.Series([True]*len(df), index=df.index)
        mask1 = matchall if seg.street_nam is None else df.f_cross_nam == seg.street_nam_upper
        mask2 = matchall if seg.street_typ is None else df.f_cross_typ == seg.street_typ_upper
        mask4 = matchall if seg.street_nam is None else df.t_cross_nam == seg.street_nam_upper
        mask5 = matchall if seg.street_typ is None else df.t_cross_typ == seg.street_typ_upper
        mask = (mask1 & mask2) | (mask4 & mask5) 
        candidates = df[mask][df[mask].intersects(seg.geometry)]
        points = candidates.intersection(seg.geometry)
        candidates = candidates.filter(like='fullname')
        seg_names = [v for k,v in seg._asdict().items() if 'fullname' in k]
        for seg_name in seg_names:
            for candidate,p in zip(candidates.itertuples(index=False), points):
                for cross_name in candidate:
                    crosses[seg_name + " and " + cross_name] = p
    crosses = pd.DataFrame.from_records(crosses.items(), columns=['streets','point'])
    crosses = gpd.GeoDataFrame(crosses, geometry='point', crs=df.crs)
    crosses.to_parquet(out_file)
    return crosses

def map_intersections(street_intersection_file, comm_area_file, out_file):
    df = gpd.read_parquet(street_intersection_file)
    comm_areas = gpd.read_parquet(comm_area_file)
    
    df = df.sjoin(comm_areas[['community_name','geometry']],
                    how='inner',
                    predicate='intersects',
                    lsuffix='intersection',
                    rsuffix='comm').drop(columns=['index_comm','geometry'])
    
    df = (df.melt(id_vars=['community_name'],
                   value_vars=['block_name1', 'block_name2', 'block_name3', 'block_name4'],
                   var_name='exp',
                   value_name='block_name')
                .drop(columns=['exp'])
                .drop_duplicates())
    
    df[['block_name','community_name']].to_parquet(out_file)
    return df