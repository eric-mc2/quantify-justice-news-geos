from functools import partial
import dagster as dg
import os

from scripts.utils.config import Config
from scripts.geoms import operations as ops
from scripts.utils.dagster import dg_standard_table

config = Config()
dg_asset = partial(dg.asset, key_prefix=["geoms"], group_name="geoms")

@dg_asset()
def comm_areas():
    in_file = config.get_data_path("raw.comm_areas")
    out_file = config.get_data_path("geoms.comm_areas")
    df = ops.clean_comm_areas(in_file, out_file)
    return dg_standard_table(df)

@dg_asset()
def street_segs():
    in_file = config.get_data_path("raw.street_segs")
    out_file = config.get_data_path("geoms.street_segs")
    df = ops.clean_street_segs(in_file, out_file)
    return dg_standard_table(df)

@dg_asset()
def street_names():
    in_file = config.get_data_path("raw.street_names")
    out_file = config.get_data_path("geoms.street_names")
    df = ops.clean_street_names(in_file, out_file)
    return dg_standard_table(df)

@dg_asset(deps=[street_segs])
def blocks():
    in_file = config.get_data_path("geoms.street_segs")
    out_file = config.get_data_path("geoms.street_blocks")
    df = ops.create_blocks(in_file, out_file)
    return dg_standard_table(df)

@dg_asset(deps=[street_segs])
def intersections():
    in_file = config.get_data_path("geoms.street_segs")
    out_file = config.get_data_path("geoms.street_intersections")
    df = ops.create_intersections(in_file, out_file)
    return dg_standard_table(df)

@dg_asset(deps=[street_segs])
def intersection_points():
    in_file = config.get_data_path("geoms.street_segs")
    out_file = config.get_data_path("geoms.street_intersection_points")
    df = ops.create_intersection_geoms(in_file, out_file)
    return dg_standard_table(df)

@dg_asset()
def hospitals():
    out_file = config.get_data_path("geoms.hospitals")
    df = ops.get_hospitals(out_file)
    return dg_standard_table(df)

@dg_asset()
def landmarks():
    out_file = config.get_data_path("geoms.landmarks")
    df = ops.get_landmarks(out_file)
    return dg_standard_table(df)

@dg_asset()
def neighborhoods():
    in_file = config.get_data_path("raw.neighborhoods")
    out_file = config.get_data_path("geoms.neighborhoods")
    df = ops.clean_neighborhoods(in_file, out_file)
    return dg_standard_table(df)

@dg_asset(deps=[neighborhoods])
def neighborhood_labels():
    in_file = config.get_data_path("geoms.neighborhoods")
    comm_area_file = config.get_data_path("geoms.comm_areas")
    out_file = config.get_data_path("geoms.neighborhood_labels")
    df = ops.neighborhood_labels(in_file,comm_area_file, out_file)
    return dg_standard_table(df)

@dg_asset(deps=[blocks, comm_areas])
def block_labels():
    street_block_file = config.get_data_path("geoms.street_blocks")
    comm_area_file = config.get_data_path("geoms.comm_areas")
    out_file = config.get_data_path("geoms.block_labels")
    df = ops.block_labels(street_block_file, comm_area_file, out_file)
    return dg_standard_table(df)

@dg_asset(deps=[intersection_points, comm_areas])
def intersection_labels():
    int_file = config.get_data_path("geoms.street_intersection_points")
    comm_area_file = config.get_data_path("geoms.comm_areas")
    out_file = config.get_data_path("geoms.intersection_labels")
    df = ops.intersection_labels(int_file, comm_area_file, out_file)
    return dg_standard_table(df)

defs = dg.Definitions(assets=[comm_areas,
                              street_segs,
                              street_names,
                              blocks,
                              intersections,
                              intersection_points,
                              hospitals,
                              landmarks,
                              neighborhoods,
                              block_labels,
                              intersection_labels,
                              neighborhood_labels])
