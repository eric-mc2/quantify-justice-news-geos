from dagster import Definitions
from scripts.prior_model import assets as pre_assets
from scripts.art_relevance import assets as art_assets
from scripts.sent_relevance import assets as sent_assets
from scripts.geoms import assets as geom_assets
from scripts.entity_recognition import assets as er_assets
from scripts.neighborhood_clf import assets as nc_assets
# from dagster import build_column_schema_change_checks

defs = Definitions.merge(pre_assets.defs,
                         art_assets.defs,
                         sent_assets.defs,
                         geom_assets.defs,
                         er_assets.defs,
                         nc_assets.defs)

# schema_checks = build_column_schema_change_checks(assets=[pre_assets.defs.assets,
#                                                           art_assets.defs.assets,
#                                                             sent_assets.defs.assets,
#                                                             geom_assets.defs.assets,
#                                                             er_assets.defs.assets])