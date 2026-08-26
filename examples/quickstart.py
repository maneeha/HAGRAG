from hagrag.config import load_config
from hagrag.runtime import load_query_engine


config = load_config("configs/paper.yaml")
engine = load_query_engine(config, algorithm="leiden", weighting="abstract")
result = engine.query("What factors are associated with type 2 diabetes management?")
print(result["response"])
