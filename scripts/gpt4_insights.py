import openai
import yaml

def get_gpt4_insights(prompt):
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def main():
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    openai.api_key = config['openai_api_key']

    prompt = "Provide insights on the recent gold futures price predictions based on the following data: ..."

    insights = get_gpt4_insights(prompt)
    with open('data/processed/gpt4_insights.txt', 'w') as file:
        file.write(insights)

if __name__ == "__main__":
    main()
