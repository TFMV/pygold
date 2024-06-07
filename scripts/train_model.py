import mage_ai
from mage_ai.data_preparation.repo_manager import get_repo_path
import yaml

def main():
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    repo_path = get_repo_path()
    project = mage_ai.data_preparation.models.Project.create(project_name='pygold', repo_path=repo_path)
    pipeline = mage_ai.data_preparation.models.Pipeline.create('gold_futures_pipeline', project_uuid=project.uuid)
    block = pipeline.add_block('transformer', 'gold_prices_transformer')

    block.execute_sync()

if __name__ == "__main__":
    main()
