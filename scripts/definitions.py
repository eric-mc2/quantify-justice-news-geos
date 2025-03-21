from dagster import Definitions
from scripts.art_relevance import assets as art_assets
from scripts.sent_relevance import assets as sent_assets
from scripts.geoms import assets as geom_assets
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

defs = Definitions.merge(art_assets.defs,
                         sent_assets.defs,
                         geom_assets.defs)
