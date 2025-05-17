import geopandas as gpd
from shapely import from_wkt
import pandas as pd
import overpass
from shapely.geometry import shape

from scripts.utils.logging import setup_logger

logger = setup_logger(__name__)
NAD_27_ILLINOIS_EAST = "EPSG:26771"

def clean_comm_areas(in_file, out_file) -> gpd.GeoDataFrame:
    df = pd.read_csv(in_file)
    df['geometry'] = df['the_geom'].apply(from_wkt)
    assert df['AREA_NUMBE'].nunique() == len(df)
    assert all(df['AREA_NUM_1'] == df['AREA_NUMBE'])
    df['id'] = df['AREA_NUMBE']
    df = df.filter(['id','geometry','COMMUNITY'])
    df = df.rename(columns={'COMMUNITY':'community_name'})
    df = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
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
    df = gpd.GeoDataFrame(street_blocks, geometry='geometry', crs=street_segs.crs)
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

def clean_neighborhoods(in_file, out_file) -> pd.DataFrame:
    df = pd.read_csv(in_file)
    names = df.melt(id_vars='the_geom', value_vars=['PRI_NEIGH','SEC_NEIGH'], value_name='name')
    names['name'] = names['name'].str.title()
    names = names.drop_duplicates()
    names['geometry'] = names['the_geom'].apply(from_wkt)
    names = gpd.GeoDataFrame(names, geometry='geometry', crs="EPSG:4326")
    names[['name','geometry']].to_parquet(out_file)
    return names

def neighborhood_labels(in_file, comm_area_file, out_file) -> pd.DataFrame:
    df = gpd.read_parquet(in_file).to_crs(NAD_27_ILLINOIS_EAST)
    comm_areas = gpd.read_parquet(comm_area_file).to_crs(NAD_27_ILLINOIS_EAST)

    intersection = df.sjoin(comm_areas[['community_name','geometry']],
                    how='left', predicate='intersects')
    touches = df.sjoin(comm_areas[['community_name','geometry']],
                    how='left', predicate='touches')
    # TODO: Normalize this so multi-communities sum to 1!
    df = intersection.merge(touches, on=['name','community_name'], how='left', indicator='_mask')
    df = df.loc[df['_mask'] == 'left_only']
    df = df.filter(['name','community_name'])
    df.to_csv(out_file, index=False)
    return df
    
def block_labels(street_block_file, comm_area_file, out_file):
    # TODO: Normalize this so multi blocks sum to 1!
    df = gpd.read_parquet(street_block_file).to_crs(NAD_27_ILLINOIS_EAST)
    comm_areas = gpd.read_parquet(comm_area_file).to_crs(NAD_27_ILLINOIS_EAST)
    
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
    
    df = df[['block_name','community_name']]
    df.to_parquet(out_file)
    return df

def create_intersection_geoms(in_file, out_file):
    df = gpd.read_parquet(in_file).to_crs(NAD_27_ILLINOIS_EAST)
    valid = (df.f_cross.str.count(r'\|') == 4) & (df.t_cross.str.count(r'\|') == 4)
    logger.debug("Dropping %d rows with invalid cross names", sum(~valid))
    
    df = (df.loc[valid]
          .assign(street_nam_upper = df.street_nam.str.upper(),
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
                   fullname9 = df.street_nam.fillna('')))
    
    fullname_cols = df.filter(regex=r'fullname\d+').columns.tolist()
    keep_left = ['street_nam','street_nam_upper','street_typ_upper', 'geometry'] + fullname_cols
    keep_right = ['f_cross_nam','f_cross_typ','geometry'] + fullname_cols
    candidates_f = (df[keep_left].merge(df[keep_right], how='inner', 
                            left_on=['street_nam_upper','street_typ_upper'],
                            right_on=['f_cross_nam','f_cross_typ'])
                    .rename(columns={'f_cross_nam':'cross_nam', 'f_cross_typ':'cross_typ'}))
    
    keep_right = ['t_cross_nam','t_cross_typ','geometry'] + fullname_cols
    candidates_t = (df[keep_left].merge(df[keep_right], how='inner', 
                            left_on=['street_nam_upper','street_typ_upper'],
                            right_on=['t_cross_nam','t_cross_typ'])
                    .rename(columns={'t_cross_nam':'cross_nam', 't_cross_typ':'cross_typ'}))
    
    candidates = pd.concat([candidates_f, candidates_t])    
    candidates = candidates.dropna(subset=['street_nam','geometry_x','geometry_y'])
    candidates = candidates.loc[candidates.geometry_x.intersects(candidates.geometry_y)]
    # Checking intersections is O(n) whereas dupe check is O(nlogn) so filter first.
    candidates = candidates.loc[candidates.index.drop_duplicates()]
    
    points = candidates.geometry_x.intersection(candidates.geometry_y).rename('point')
    cross_left = candidates.filter(regex=r"fullname\d+_x").melt(var_name='variant', value_name='fullname', ignore_index=False)
    cross_right = candidates.filter(regex=r"fullname\d+_y").melt(var_name='variant', value_name='fullname', ignore_index=False)
    crosses = cross_left.join(cross_right, how='inner', lsuffix='_x', rsuffix='_y').join(points, how='inner')
    crosses['cross_name'] = crosses['fullname_x'] + " and " + crosses['fullname_y']
    crosses = gpd.GeoDataFrame(crosses[['cross_name','point']], geometry='point', crs=df.crs)
    crosses = crosses.to_crs("EPSG:4326")
    crosses.to_parquet(out_file)
    return crosses

def intersection_labels(street_intersection_file, comm_area_file, out_file):
    # TODO: Normalize this too!
    df = gpd.read_parquet(street_intersection_file).to_crs(NAD_27_ILLINOIS_EAST)
    comm_areas = gpd.read_parquet(comm_area_file).to_crs(NAD_27_ILLINOIS_EAST)
    
    df = df.sjoin(comm_areas[['community_name','geometry']],
                    how='inner',
                    predicate='intersects')
    
    df = df[['cross_name','community_name']]
    df.to_parquet(out_file)
    return df