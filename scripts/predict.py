import yaml
import mage_ai
from mage_ai.data_preparation.repo_manager import get_repo_path

def main():
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    repo_path = get_repo_path()
    pipeline = mage_ai.data_preparation.models.Pipeline.get('gold_futures_pipeline', repo_path)
    pipeline.run()

if __name__ == "__main__":
    main()
