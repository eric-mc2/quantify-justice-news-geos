from dagster import Definitions
from scripts.art_relevance import assets as art_assets
from scripts.sent_relevance import assets as sent_assets
from scripts.geoms import assets as geom_assets
from scripts.entity_recognition import assets as er_assets

defs = Definitions.merge(art_assets.defs,
                         sent_assets.defs,
                         geom_assets.defs,
                         er_assets.defs)
