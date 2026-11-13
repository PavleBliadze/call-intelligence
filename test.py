"""Quick check that the OPENAI_API_KEY in .env works."""

from openai import OpenAI

from config import OPENAI_API_KEY, TEXT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.chat.completions.create(
    model=TEXT_MODEL,
    messages=[{"role": "user", "content": "Say hello!"}],
)

print("API key works. Response:")
print(response.choices[0].message.content)
